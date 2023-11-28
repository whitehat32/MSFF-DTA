# -*- encoding: utf-8 -*-
'''
@Time       : 2022/11/26 17:19
@Author     : Tian Tan
@Email      :
@File       : gnn.py
@Project    :
@Description:
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor
from torch_geometric.nn import GINEConv as BaseGINEConv, GINConv as BaseGINConv


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False, dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y


class GINConv(BaseGINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten
        else:
            return x_j


class GIN(nn.Module):
    def __init__(self, hid_dim, dropout=0.0, n_layers=3, use_edge_attr=False, edge_attr_dim=0):
        super().__init__()

        self.n_layers = n_layers
        hidden_size = hid_dim
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = dropout
        self.use_edge_attr = False

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        if edge_attr_dim != 0 and self.use_edge_attr:
            self.edge_encoder = Linear(edge_attr_dim, hidden_size)
        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(self.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(self.MLP(hidden_size, hidden_size)))



    def forward(self, x, edge_index, edge_attr=None, edge_atten=None, concat=False):
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        if concat:
            x_list = []

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            if concat:
                x_list.append(x)
        if concat:
            return x_list
        return x

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )


class GINEConv(BaseGINEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        m = (x_j + edge_attr).relu()

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m

class ResGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResGCN, self).__init__()
        #print(in_dim)
        #print(out_dim)
        self.res_conv1 = GraphConv(in_dim, out_dim)
        self.res_conv2 = GraphConv(out_dim, out_dim)
        self.res_conv3 = GraphConv(out_dim, out_dim)      

    def forward(self, inputs, adj_1hop=None, adj_2hop=None, adj_3hop=None, flag=0):
        # print(adj)
        outputs1 = self.res_conv1(inputs, adj_1hop)
        outputs2 = self.res_conv2(outputs1, adj_1hop)
        outputs3 = self.res_conv3(outputs2, adj_1hop)

        if flag:
            # subgraph1 = torch.bmm(adj_1hop, outputs1)
            # subgraph2 = torch.bmm(adj_2hop, outputs2)
            subgraph3 = torch.bmm(adj_3hop, outputs3)
            # outputs1 = torch.cat([outputs1, subgraph1], dim=-1)
            # outputs2 = torch.cat([outputs2, subgraph2], dim=-1)
            outputs3 = torch.cat([outputs3, subgraph3], dim=-1)

        # return outputs3
        return torch.cat([outputs1, outputs2, outputs3], dim=-1)


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleGCN, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, inputs, adj):
        return torch.bmm(adj, self.fc(inputs))


class MultiGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiGCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)

        # self.fc2 = nn.Linear(out_dim, out_dim, bias=True)
        # self.fc3 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, inputs, adj):
        # outputs = self.act(torch.bmm(adj, self.fc1(inputs)))
        outputs = torch.bmm(adj, self.fc1(inputs))
        # outputs = torch.bmm(adj, self.fc2(outputs))
        # outputs = torch.bmm(adj, self.fc3(outputs))
        return outputs


class MultiGCN_cp(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(MultiGCN_cp, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=True)
        self.fc3 = nn.Linear(out_dim, out_dim, bias=True)
        
        self.sub_fc = nn.Linear(out_dim, out_dim, bias=True)
        

    def forward(self, inputs, adj):
        outputs1 = torch.bmm(adj, self.fc1(inputs))
        outputs2 = torch.bmm(adj, self.fc2(outputs1))
        outputs3 = torch.bmm(adj, self.fc3(outputs2))
        
        #adj_sub = adj @ adj @ adj
        #subgraph3 = self.sub_fc(torch.bmm(adj_sub, outputs3))
	
        return torch.stack([outputs1, outputs2, outputs3], dim=0)
        # return outputs3
        # 0.4662 0.4684
class GATLayer(nn.Module):
    """GAT层"""
    def __init__(self, input_feature, output_feature, dropout, alpha, concat=False):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))
        self.w = nn.Linear(input_feature, output_feature, bias=True)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):


        # w = self.w
        # print(w.shape)
        # print(h.shape)
        Wh = self.w(h)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # adj>0的位置使用e对应位置的值替换，其余都为-9e15，这样设定经过Softmax后每个节点对应的行非邻居都会变为0。
        attention = F.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)  # 得到下一层的输入

        if self.concat:
            return F.elu(h_prime)  # 激活
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # N*out_size @ out_size*1 = N*1

        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # N*1

        e = Wh1 + Wh2.transpose(1, 2)  # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e)


class GAT(nn.Module):
    """GAT模型"""
    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, nheads, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in
                          range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(hidden_size * nheads, output_size, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attention], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))

        # return F.log_softmax(x, dim=1)
        return x

class MultiGAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, nheads, concat=True):
        super(MultiGAT, self).__init__()
        self.layers = nn.ModuleList([GAT(input_size, hidden_size, output_size, dropout, alpha, nheads) for _ in range(3)])

    def forward(self, inputs, adj):
        for layer in self.layers:
            inputs = layer(inputs, adj)
        outputs = inputs
        return outputs
