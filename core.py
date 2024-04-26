# -*- encoding: utf-8 -*-
'''
@Time       : 2022/11/26 17:19
@Author     : Tian Tan
@Email      :
@File       : core.py
@Project    :
@Description:
'''

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from einops.layers.torch import Rearrange
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch, softmax, scatter

from data import CPIDataset
from model.gnn import GIN
from model.mainModel import DPI, MFB

from preprocessing.protein import ProteinFeatureManager
from preprocessing.compound import CompoundFeatureManager
from dc1d.nn import gLN

from optim.lion_pytorch import Lion
from data import CPIDataset_case


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    y = np.array(y)
    f = np.array(f)
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def DiscreteEncoder(nin, nhid):
    return nn.Embedding(nin, nhid)


def LinearEncoder(nin, nhid):
    return nn.Linear(nin, nhid)


def FeatureEncoder(TYPE, nin, nhid):
    models = {
        'Discrete': DiscreteEncoder,
        'Linear': LinearEncoder
    }
    return models[TYPE](nin, nhid)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch=None):
        for module in self._modules.values():
            # if isinstance(module, (InstanceNorm)):
            #     inputs = module(inputs)
            # else:
            inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout=0.2, bias=True):

        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                # m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))
        super(MLP, self).__init__(*m)


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, d=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=d),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)
    


class PackedConv1d(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = "same",
        padding_mode: str = "reflect",
        offset_groups: int = 1,
        *args,
        **kwargs
        ) -> None:
        """
        Packed 1D Deformable convolution class. Depthwise-Separable convolution is used to compute offsets.
        Args:
            in_channels (int): Value of convolution kernel size
            out_channels (int): Value of convolution kernel dilation factor
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int): See torch.nn.Conv1d for details. Default "valid". Still experimental beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int): 1 or in_channels
            bias (bool): Whether to use bias. Default = True
            padding_mode (str): See torch.nn.Conv1d for details. Default "reflect". Still experimental beware of unexpected behaviour.
            offset_groups (int): 1 or in_channels
            device: Device to operate function on. Default: torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """
        assert offset_groups in [1,in_channels], "offset_groups only implemented for offset_groups in {1,in_channels}"
        super(PackedConv1d, self).__init__(*args,**kwargs)
        # super(PackedDeformConv1d,self).__init__(
        #     in_channels = in_channels,
        #     out_channels = out_channels,
        #     kernel_size = kernel_size,
        #     stride = stride,
        #     padding = padding,
        #     dilation = dilation,
        #     groups = groups,
        #     bias = bias,
        #     padding_mode = padding_mode,
        #     device = device,
        #     # interpolation_function = interpolation_function,
        #     unconstrained=unconstrained,
        #     *args,
        #     **kwargs
        #     )
        self.offset_groups = offset_groups

        self.offset_dconv = nn.Conv1d(in_channels,in_channels,kernel_size,stride=1,groups=in_channels,padding=padding,padding_mode=padding_mode,bias=False)
        self.odc_norm = gLN(in_channels)
        self.odc_prelu = nn.PReLU()
        self.offset_pconv = nn.Conv1d(in_channels,out_channels,1,stride=1,bias=False)
        self.odp_norm = gLN(kernel_size*offset_groups)
        self.odp_prelu = nn.PReLU()
    
    def forward(self, input, with_offsets=False):
        """
        Forward pass of 1D deformable convolution layer
        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor
            
        Returns:
            output (Tensor[batch_size, in_channels, length]): output tensor
        """
        offsets = self.offset_dconv(input)
        offsets = self.odc_norm(self.odc_prelu(offsets).moveaxis(1,2)).moveaxis(2,1)
        offsets = self.offset_pconv(offsets)
        # print('shape compare = ', offsets.shape, input.shape)
        return offsets




class DeformConv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, d=1):
        super().__init__()
        # padding = int((kernel_size - 1) / 2) * d
        self.inc = nn.Sequential(
            PackedConv1d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                offset_groups=in_channels
            ),
            # nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    #   padding=padding, dilation=d),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)




class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, d=[1, 1, 1], dropout=0.2):
        super().__init__()

        # self.inc = nn.Sequential(OrderedDict([('conv_layer0',
        #                                        Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size[0],
        #                                                   stride=stride, d=d[0]))]))
        #
        # for layer_idx in range(layer_num - 1):
        #     self.inc.add_module('conv_layer%d' % (layer_idx + 1),
        #                         Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size[layer_idx + 1],
        #                                    stride=stride, d=d[layer_idx + 1]))
        self.cnn_layers = torch.nn.ModuleList([Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size[0],
                                                          stride=stride, d=d[0])])
        for layer_idx in range(layer_num - 1):
            self.cnn_layers.append(Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size[layer_idx + 1],
                                              stride=stride, d=d[layer_idx + 1]))
        # self.do = torch.nn.Dropout(dropout)
        # self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        return x
    

class StackDeformCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, d=3, dropout=0.2):
        super().__init__()

        # self.inc = nn.Sequential(OrderedDict([('conv_layer0',
        #                                        Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size[0],
        #                                                   stride=stride, d=d[0]))]))
        #
        # for layer_idx in range(layer_num - 1):
        #     self.inc.add_module('conv_layer%d' % (layer_idx + 1),
        #                         Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size[layer_idx + 1],
        #                                    stride=stride, d=d[layer_idx + 1]))


        self.cnn_layers = torch.nn.ModuleList([DeformConv1dReLU(in_channels, out_channels, kernel_size=kernel_size[0],
                                                          stride=stride, d=d)])
        for layer_idx in range(layer_num - 1):
            self.cnn_layers.append(DeformConv1dReLU(in_channels, out_channels, kernel_size=kernel_size[0],
                                                          stride=stride, d=d))
        # self.do = torch.nn.Dropout(dropout)
        # self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        return x


# class CDilated(nn.Module):
#
#     def __init__(self, nIn, nOut, kSize, stride=1, d=1):
#         super().__init__()
#         padding = int((kSize - 1) / 2) * d
#         self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)
#
#     def forward(self, input):
#         output = self.conv(input)
#         return output
#
#
# class DilatedParllelResidualBlockB(nn.Module):
#
#     def __init__(self, nIn, nOut, add=True):
#         super().__init__()
#         n = int(nOut / 4)
#         n1 = nOut - 3 * n
#         self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
#         self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
#         self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
#         self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
#         self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
#         self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
#         self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
#
#         if nIn != nOut:
#             add = False
#         self.add = add
#
#     def forward(self, input):
#         # reduce
#         output1 = self.c1(input)
#         output1 = self.br1(output1)
#         # split and transform
#         d1 = self.d1(output1)
#         d2 = self.d2(output1)
#         d4 = self.d4(output1)
#         d8 = self.d8(output1)
#
#         # heirarchical fusion for de-gridding
#         add1 = d2
#         add2 = add1 + d4
#         add3 = add2 + d8
#
#         # merge
#         combine = torch.cat([d1, add1, add2, add3], 1)
#
#         # if residual version
#         if self.add:
#             combine = input + combine
#         output = self.br2(combine)
#         return output


# class CNN_MLP(nn.Module):
#     def __init__(self, Affine, patch, channel, output_size, down=False, last=True):
#         super(CNN_MLP, self).__init__()
#         self.Afine_p1 = Affine(channel)
#         self.Afine_p2 = Affine(channel)
#         self.Afine_p3 = Affine(channel)
#         self.Afine_p4 = Affine(channel)
#         self.cross_patch_linear0 = nn.Linear(patch, patch)
#         self.cross_patch_linear1 = nn.Linear(patch, patch)
#         self.cross_patch_linear = nn.Linear(patch, patch)
#         self.cnn1 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=7, padding=3, groups=patch)
#         self.cnn2 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=7, padding=3, groups=patch)
#         self.cnn3 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=7, padding=3, groups=patch)
#         self.attention_channel_linear2 = nn.Linear(channel, channel)
#         self.last_linear = nn.Linear(channel, output_size)
#         self.bnp = nn.BatchNorm1d(patch)
#         self.act = nn.ReLU()
#         self.last = last
#         self.dropout = nn.Dropout(0.05)
#         self.down = down
#
#     def forward(self, x):
#         # print(x.shape)
#         x_cp = self.Afine_p1(x).permute(0, 2, 1)
#         x_cp = self.act(self.cross_patch_linear0(x_cp))
#         x_cp = self.act(self.cross_patch_linear1(x_cp))
#         x_cp = self.cross_patch_linear(x_cp).permute(0, 2, 1)
#         x_cc = x + self.Afine_p2(x_cp)
#         x_cc2 = self.Afine_p3(x_cc)
#         x_cc2 = self.act(self.cnn1(x_cc2))
#         x_cc2 = self.act(self.cnn2(x_cc2))
#         x_cc2 = self.act(self.cnn3(x_cc2))
#         x_cc2 = self.Afine_p4(x_cc2)
#
#         if self.last == True:
#             x_out = self.last_linear(x_cc2)
#         return x_out


# class Stack_CNN_MLP(nn.Module):
#     def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=1):
#         super(Stack_CNN_MLP, self).__init__()
#
#         self.inc = nn.Sequential(OrderedDict([('conv_layer0',
#                                                CNN_MLP(Affine, 1200, 100, 75, 0,True)]))
#         for layer_idx in range(layer_num - 1):
#             self.inc.add_module('conv_layer%d' % (layer_idx + 1),
#                                 Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                                            padding=padding))
#
#
#     def forward(self, x):
#         # print(x.shape)
#
#         return x

# class Affine(nn.Module):
#     def __init__(self, dim):
#         super(Affine, self).__init__()
#         self.g = nn.Parameter(torch.ones(1, 1, dim))
#         self.b = nn.Parameter(torch.zeros(1, 1, dim))
#
#     def forward(self, x):
#         return x * self.g + self.b

def concrete_sample(att_log_logit, temp, training):
    # random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
    # random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
    # att_bern = ((att_log_logit + random_noise) / temp).sigmoid()

    if training:
        random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
    else:
        att_bern = (att_log_logit).sigmoid()
    return att_bern


def lift_node_att_to_edge_att(node_att, edge_index):
    src_lifted_att = node_att[edge_index[0]]
    dst_lifted_att = node_att[edge_index[1]]
    edge_att = src_lifted_att * dst_lifted_att
    return edge_att


class PatchGNN(nn.Module):
    r"""
    Args:
    ----------
    embed_dim (int):        the embeding dimension
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree
    """

    def __init__(self, hid_dim, num_layers=3,
                 batch_norm=False, concat=True, khopgnn=True, dropout=0.5, p=32):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gnns = nn.ModuleList([GIN(hid_dim, dropout, 1) for _ in range(num_layers)])
        self.hid_dim = hid_dim
        # self.gin = GIN(hid_dim, dropout, 1)
        self.do = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.reshape = Rearrange('(B p) d ->  B p d', p=p)
        self.extractor = torch.nn.Sequential(torch.nn.Linear(hid_dim, 2 * hid_dim), nn.ReLU(),
                                             nn.Dropout(), torch.nn.Linear(2 * hid_dim, hid_dim), nn.ReLU(),
                                             nn.Dropout(), torch.nn.Linear(hid_dim, 1))
        # inner_dim = (num_layers) * hid_dim if khopgnn else hid_dim

        self.U = nn.ModuleList(
            [MLP([hid_dim, hid_dim]) for _ in range(num_layers - 1)])
        self.V = nn.ModuleList(
            [MLP([hid_dim, hid_dim]) for _ in range(num_layers - 1)])
        self.out_proj = nn.Linear(hid_dim, hid_dim)

        # 由gnn选择主骨架

    def forward(self, x, data, edge_attr=None, edge_atten=None, sub_graph=True, training=True):
        # x_cat = [x]
        x = x[data.subgraphs_nodes_mapper]
        coarsen_adj = data.coarsen_adj
        # e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce='mean')
                subgraph_fus = self.reshape(subgraph)
                subgraph_fus = torch.matmul(coarsen_adj, subgraph_fus)
                subgraph_fus = subgraph_fus.view(-1, self.hid_dim)
                subgraph = F.relu(subgraph[batch_x] + self.do(self.V[i - 1](subgraph_fus[batch_x])))
                x = x + self.do(self.U[i - 1](subgraph))
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = x + gnn(x, edge_index, concat=False, edge_atten=edge_atten)
        if sub_graph:
            x_att = self.extractor(x)
            x_att = concrete_sample(x_att, 1, training)
            edge_att = lift_node_att_to_edge_att(x_att, edge_index)
            x = x * x_att
            subgraph_x = scatter(x, batch_x, dim=0, reduce='mean')
            subgraph_x = self.reshape(subgraph_x)
            x = self.out_proj(subgraph_x)
        return x, edge_att


class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        self.protein_dim = args.protein_dim
        self.atom_dim = args.atom_dim
        # self.embedding_dim = args.embedding_dim
        self.hid_dim = args.hid_dim
        self.mol2vec_embedding_dim = args.mol2vec_embedding_dim
        self.protein_reshape = Rearrange('(B p) d ->  B p d', p=32)
        ####################
        # self.pretrain_emb_fc = nn.Linear(self.args.protein_embedding_dim, self.hid_dim * 4)
        # self.concat_fc = nn.Linear(self.hid_dim * 4, self.hid_dim)
        # self.concat_ln = nn.LayerNorm(self.hid_dim)
        self.protein_pool = nn.AdaptiveMaxPool1d(1)
        self.protein_fg_emb = nn.Embedding(217, self.hid_dim, padding_idx=0)
        # self.mix = MixerBlock(dim=self.hid_dim, num_patch=1200, token_dim=self.hid_dim, channel_dim=self.hid_dim, dropout=0.2)

        self.block_list = nn.ModuleList()
        self.block_list_sub = nn.ModuleList()
        self.block_list2 = nn.ModuleList()
        self.block_list3 = nn.ModuleList()

        for block_idx in range(3):
            self.block_list.append(
                StackCNN(block_idx, self.hid_dim, self.hid_dim, kernel_size=[15, 15, 15], d=[1, 1, 1])
                # StackDeformCNN(block_idx, self.hid_dim, self.hid_dim, kernel_size=[15, 15, 15], d=3)
            )

        for block_idx in range(3):
            self.block_list_sub.append(
                # StackDeformCNN(block_idx, self.hid_dim, self.hid_dim, kernel_size=[15, 15, 15], d=3)
                StackCNN(block_idx, self.hid_dim, self.hid_dim, kernel_size=[3, 3, 3], d=[1, 1, 1])
            )

        self.decoder = DPI(
            atom_dim=args.atom_dim,
            fp_dim=args.compound_embedding_dim,
            hid_dim=args.hid_dim,
            drug_n_layers=args.decoder_layers,
            drug_n_heads=args.n_heads,
            pf_dim=args.pf_dim,
            dropout=args.dropout,
            objective=args.objective,
            tape_emb_dim=args.hid_dim * 5
            # mol2vec_embedding_dim=self.mol2vec_embedding_dim
        )
        self.protein_fusion = MFB(self.hid_dim * 3, self.hid_dim * 3, self.hid_dim * 3, k=2)
        self.protein_input = nn.Linear(21, self.hid_dim)
        # self.protein_readout1 = nn.Linear(self.hid_dim * 3, self.hid_dim)
        # self.protein_readout2 = nn.Linear(self.hid_dim * 3, self.hid_dim)
        # self.protein_readout3 = nn.Linear(self.hid_dim * 3, self.hid_dim)
        self.protein_readout = nn.Linear(self.hid_dim * 3, self.hid_dim)
        # self.protein_fc1 = nn.Linear(self.hid_dim, self.hid_dim*3)
        # self.protein_fc2 = nn.Linear(self.hid_dim, self.hid_dim*3)
        self.protein_sub_readout = nn.Linear(self.hid_dim, self.hid_dim)
        self.do = nn.Dropout()
        self.objective = args.objective
        # self.act = torch.nn.ReLU()
        self.act = torch.nn.GELU()

    def get_param(self, *shape):
        param = nn.Parameter(torch.zeros(shape))
        nn.init.xavier_normal_(param)
        return param

    def make_masks_ori(self, atom_num, protein_num, compound_max_len, protein_max_len):
        batch_size = len(atom_num)
        compound_mask = torch.zeros((batch_size, compound_max_len)).type_as(atom_num)
        protein_mask = torch.zeros((batch_size, protein_max_len)).type_as(atom_num)

        for i in range(batch_size):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return compound_mask, protein_mask

    def make_masks(self, protein_num, protein_max_len):
        batch_size = len(protein_num)
        protein_mask = torch.zeros((batch_size, protein_max_len)).type_as(protein_num)

        for i in range(batch_size):
            protein_mask[i, :protein_num[i]] = 1
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return protein_mask

    # def make_masks(self, atom_num, compound_max_len):
    #     batch_size = len(atom_num)
    #     compound_mask = torch.zeros((batch_size, compound_max_len)).type_as(atom_num)
    #     sub_graph_mask = torch.zeros((batch_size, 1)).type_as(atom_num)
    #     for i in range(batch_size):
    #         compound_mask[i, :atom_num[i]] = 1
    #         # sub_graph_mask[i, :atom_num[i]] = 1
    #     compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)
    #     sub_graph_mask = sub_graph_mask.unsqueeze(1).unsqueeze(2)
    #     return compound_mask, sub_graph_mask

    def forward(self, batch):
        # protein
        protein_fg = batch['PROTEIN_FG']
        # protein_fg2 = batch['PROTEIN_FG2']
        # protein_fg3 = batch['PROTEIN_FG3']

        protein_fg = self.protein_fg_emb(protein_fg)
        # protein_fg2 = self.protein_fg_emb(protein_fg2)
        # protein_fg3 = self.protein_fg_emb(protein_fg3)
        protein_num = batch["PROTEIN_NUMS"]

        protein_input = self.protein_input(batch["PROTEIN_NODE_FEAT"])
        protein_mask = self.make_masks(protein_num - 2, protein_fg.shape[1])

        protein_input = protein_input.permute(0, 2, 1)
        protein_sub_list = [block(protein_input).permute(0, 2, 1) for block in self.block_list_sub]
        protein_sub = torch.cat(protein_sub_list, dim=-1)
        # protein_sub = self.protein_fc1(protein_sub_list[-1]) 



        # protein_mask = None

        # compound
        compound_graph = batch["COMPOUND_GRAPH"]
        compound = compound_graph.x
        degree = batch["COMPOUND_DEGREE"]
        compound_fp = batch["COMPOUND_FP"]
        compound_pos = batch['COMPOUND_POS']
        # compound_pos = None

        # protein_cnn = self.cnn(protein_graph.x)

        # protein_gnn = self.patch_gnn(protein_input, protein_graph)
        # protein_gnn = global_mean_pool(torch.cat(protein_gnn, dim=-1), protein_graph.batch)

        # protein_patch, protein_edge_att = self.patch_gnn(protein_input, protein_graph, sub_graph=True, training=type)
        protein_fg = torch.cat([protein_fg, protein_fg[:,-1,:].unsqueeze(1).clone(),protein_fg[:,-1,:].unsqueeze(1).clone()], dim=1)
        protein_input = protein_fg.permute(0, 2, 1)
        protein_seq_list = [block(protein_input) for block in self.block_list]

        # protein_seq_node = [protein_seq_list[i].permute(0, 2, 1) for i in range(len(protein_seq_list))]
        protein_seq_node = [protein_seq_list[i].permute(0, 2, 1) for i in range(len(protein_seq_list))]
        # protein_seq_node = self.protein_fc2(protein_seq_node[-1])  #消融MSCNN
        protein_seq_node = torch.cat(protein_seq_node, dim=-1)

        protein_sub = self.protein_fusion(protein_seq_node, protein_sub[:, :protein_fg.shape[1]], ln=False)
        protein_seq_list = [self.protein_pool(protein_seq_list[i]).squeeze(-1) for i in range(len(protein_seq_list))]
        # protein = self.protein_readout1(torch.cat(protein_seq_list, dim=-1))

        # protein_input2 = protein_fg2.permute(0, 2, 1)
        # protein_seq_list = [self.protein_pool(block(protein_input2)).squeeze(-1) for block in self.block_list]
        # protein2 = self.protein_readout2(torch.cat(protein_seq_list, dim=-1))
        #
        # protein_input3 = protein_fg3.permute(0, 2, 1)
        # protein_seq_list = [self.protein_pool(block(protein_input3)).squeeze(-1) for block in self.block_list]
        # protein3 = self.protein_readout3(torch.cat(protein_seq_list, dim=-1))

        # protein = self.act(self.protein_readout(torch.cat(protein_seq_list, dim=-1)))
        protein = None
        # 蛋白质片段
        # protein_fg = protein_fg.permute(0, 2, 1)
        # protein_subseq = self.protein_subseq_readout(self.cnn(protein_fg).permute(0, 2, 1))
        # 蛋白质片段

        out = self.decoder(compound, compound_fp, protein_sub, compound_graph, protein_num, compound_pos=compound_pos)
        return out[0], out[1], out[2]


class DeepCPIModel(pl.LightningModule):
    """DeepCPIModel-Lightning模型

    Args:
        args: 模型参数, 参见main.py文件
    """

    def __init__(self, args):
        super(DeepCPIModel, self).__init__()
        self.args = args
        self.root_data_path = args.root_data_path

        self.predictor = Predictor(args)

        self.protein_feature_manager = ProteinFeatureManager(self.root_data_path)
        self.compound_feature_manager = CompoundFeatureManager(self.root_data_path)
        
        self.train_set = CPIDataset(os.path.join(self.root_data_path, "train_fold0.csv"),
                                    self.protein_feature_manager, self.compound_feature_manager,
                                    self.args)

        # self.train_set = CPIDataset_case(os.path.join(self.root_data_path, "case_study_true.csv"),
        #                            self.protein_feature_manager, self.compound_feature_manager,
        #                            self.args)

        print('The train data number is {}'.format(len(self.train_set)))

        self.valid_set = CPIDataset(os.path.join(self.root_data_path, "test_fold0.csv"),
                                    self.protein_feature_manager, self.compound_feature_manager,
                                    self.args)
        print('The valid data number is {}'.format(len(self.valid_set)))

        self.test_set = CPIDataset(os.path.join(self.root_data_path, "test_fold0.csv"),
                                   self.protein_feature_manager, self.compound_feature_manager,
                                   self.args)
        print('The test data number is {}'.format(len(self.test_set)))

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if self.root_data_path == "data/Davis":
            weight_CE = torch.FloatTensor([0.3, 0.7]).cuda()
        elif self.root_data_path == "data/KIBA":
            weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()
        else:
            weight_CE = None

        if args.objective == "classification":
            self.criterion = nn.CrossEntropyLoss(weight_CE)
        elif args.objective == "regression":
            self.criterion = nn.MSELoss()

    def forward(self, batch):
        return self.predictor(batch)

    def predict_dataloader(self):
    # Define and return the data loader for prediction
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            collate_fn=self.train_set.collate_fn,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )



    def train_dataloader(self):
        # self.train_set.protein_feature_manager.protein_graph_dict \
        #     = self.train_set.protein_feature_manager.get_protein_graph(training=True)
        # self.train_set.get_mol_graph()
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            collate_fn=self.train_set.collate_fn,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.args.batch_size,
            collate_fn=self.valid_set.collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):

        # if self.args.mode == 'test':
        #     # print('Testing on test data')
        #     test_set = CPIDataset(os.path.join(self.root_data_path, "prot_cold_start/test_fold0.csv"),
        #                           self.protein_feature_manager, self.args)
        # else:
        #     test_set = CPIDataset(os.path.join(self.root_data_path, "prot_cold_start/test_fold0.csv"),
        #                           self.protein_feature_manager, self.args)

        return DataLoader(
            self.test_set,
            batch_size=self.args.batch_size,
            collate_fn=self.test_set.collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

    def configure_optimizers(self):
        optimizer = Lion(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        return optimizer
    
    def predict_step(self, batch, batch_idx):
        rnt, _, _ = self(batch)
        return rnt

    def training_step(self, batch, batch_idx):
        interactions = batch["LABEL"]
        if self.args.objective == "classification":
            interactions = interactions.long()
        outputs, r_att, _ = self(batch)
        
        # r_loss = torch.mean(r_att) * 0.0001
        # r_loss = r_att
        # r_loss = (r_att * torch.log(r_att / r + 1e-6) + (1 - r_att) * torch.log( (1 - r_att) / (1 - r + 1e-6) + 1e-6)).mean()
        # r_loss = torch.mean(r_att)
        
        loss = self.criterion(outputs, interactions)

        # loss = self.criterion(outputs, interactions)
        # loss = self.criterion(outputs, interactions) + (info_loss_1hop + info_loss_2hop + info_loss_3hop) / 3
        res = {"loss": loss}
        self.train_step_outputs.append(res)
        # res = {"loss": loss}
        # res = {"loss": loss}
        return res

    def on_train_epoch_end(self):
        outputs = self.train_step_outputs
        avg_loss = torch.stack([x["loss"] for x in outputs])
        avg_loss = avg_loss.mean()

        # r_loss = torch.stack([x["r_loss"] for x in outputs])
        # r_loss = r_loss.mean()
        # print('r_loss=', r_loss)
        print('loss = ', avg_loss)
        self.train_step_outputs.clear()  # free memory
        if self.args.objective == "classification":
            pass
        else:
            pass

    def validation_step(self, batch, batch_idx):
        interactions = batch["LABEL"]
        if self.args.objective == "classification":
            interactions = interactions.long()

        outputs, r_att, _ = self(batch)
        # print('r_att = ', r_att)
        loss = self.criterion(outputs, interactions)
        if self.args.objective == "classification":
            scores = F.softmax(outputs, dim=1)[:, 1].to("cpu").data.tolist()
            correct_labels = interactions.to("cpu").data.tolist()
            return {
                "val_loss": loss,
                "scores": scores,
                "correct_labels": correct_labels,
            }
        else:
            scores = outputs.to("cpu").data.tolist()
            labels = interactions.to("cpu").data.tolist()
            res = {"val_loss": loss, "scores": scores, "labels": labels}
            self.validation_step_outputs.append(res)
            return res

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs])
        avg_loss = avg_loss.mean()

        if self.args.objective == "classification":
            scores, correct_labels = [], []
            for x in outputs:
                scores.extend(x["scores"])
                correct_labels.extend(x["correct_labels"])

            auc = torch.tensor(roc_auc_score(correct_labels, scores))
            print('auc ', auc)
            self.log("auc", auc, sync_dist=True)
            self.log('val_loss', avg_loss, sync_dist=True)

        else:
            scores, correct_labels = [], []
            for x in outputs:
                scores.extend(x["scores"])
                correct_labels.extend(x["labels"])

            # rmse = np.sqrt(mean_squared_error(correct_labels, scores))
            mse = mean_squared_error(correct_labels, scores)
            print('mse ', mse)
            # print('scores =', len(scores))
            self.log("mse", mse, sync_dist=True)
            self.log("val_loss", avg_loss, sync_dist=True)
            self.validation_step_outputs.clear()

    # Prepare for rm2
    def get_k(self, y_obs, y_pred):
        y_obs = np.array(y_obs)
        y_pred = np.array(y_pred)
        return sum(y_obs * y_pred) / sum(y_pred ** 2)
    # Prepare for rm2
    def squared_error_zero(self, y_obs, y_pred):
        k = self.get_k(y_obs, y_pred)
        y_obs = np.array(y_obs)
        y_pred = np.array(y_pred)
        y_obs_mean = np.mean(y_obs)
        upp = sum((y_obs - k * y_pred) ** 2)
        down = sum((y_obs - y_obs_mean) ** 2)

        return 1 - (upp / down)
    # Prepare for rm2
    def r_squared_error(self, y_obs, y_pred):
        y_obs = np.array(y_obs)
        y_pred = np.array(y_pred)
        y_obs_mean = np.mean(y_obs)
        y_pred_mean = np.mean(y_pred)
        mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
        y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
        y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
        return mult / (y_obs_sq * y_pred_sq)


    def get_rm2(self, Y, P):
        r2 = self.r_squared_error(Y, P)
        r02 = self.squared_error_zero(Y, P)

        return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))
    def test_step(self, batch, batch_idx):
        # print('1111')
        interactions = batch["LABEL"]
        if self.args.objective == "classification":
            interactions = interactions.long()

        outputs, r_att, _ = self(batch)
        # print('r_att = ', r_att)
        loss = self.criterion(outputs, interactions)

        if self.args.objective == "classification":
            scores = F.softmax(outputs, dim=1)[:, 1].to("cpu").data.tolist()
            correct_labels = interactions.to("cpu").data.tolist()
            predict_labels = [1. if i >= 0.50 else 0. for i in scores]
            return {
                "test_loss": loss,
                "scores": scores,
                "correct_labels": correct_labels,
                "predict_labels": predict_labels,
            }
        elif self.args.objective == "regression":
            predict_values = outputs.to("cpu").data.tolist()
            correct_values = interactions.to("cpu").data.tolist()
            res = {
                "test_loss": loss,
                "predict_values": predict_values,
                "correct_values": correct_values,
            }
            self.test_step_outputs.append(res)
            return res

    def target_eval(self, gt_label, pre_label, scores, pros):
        seqs = set(pros)
        pros = np.array(pros)
        auc_t = []
        acc_t = []
        recall_t = []
        precision_t = []
        f1_score_t = []
        seq_t = []
        print('Target Test Results: Evaluate on {} proteins'.format(len(seqs)))
        for seq in seqs:
            index = pros == seq
            gt_label_t = gt_label[index]
            pre_label_t = pre_label[index]
            scores_t = scores[index]
            auc_t.append(roc_auc_score(gt_label_t, scores_t))
            acc_t.append(accuracy_score(gt_label_t, pre_label_t))
            recall_t.append(recall_score(gt_label_t, pre_label_t))
            precision_t.append(precision_score(gt_label_t, pre_label_t))
            f1_score_t.append(f1_score(gt_label_t, pre_label_t))
            seq_t.append(seq)
        auc_t = np.array(auc_t)
        acc_t = np.array(acc_t)
        recall_t = np.array(recall_t)
        precision_t = np.array(precision_t)
        f1_score_t = np.array(f1_score_t)
        seq_t = np.array(seq_t)
        print(" acc: {} (std: {}),".format(round(np.mean(acc_t), 4),
                                           round(np.std(acc_t),
                                                 4)), " auc: {}(std: {}),".format(round(np.mean(auc_t), 4),
                                                                                  round(np.std(auc_t), 4)),
              " precision: {}(std: {}),".format(round(np.mean(precision_t), 4), round(np.std(precision_t), 4)),
              " recall: {}(std: {}),".format(round(np.mean(recall_t), 4), round(np.std(recall_t), 4)),
              " f1_score: {}(std: {})".format(round(np.mean(f1_score_t), 4), round(np.std(f1_score_t), 4)))

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        # print(avg_los
        if self.args.objective == 'classification':
            scores, correct_labels, predict_labels = [], [], []
            for x in outputs:
                scores.extend(x["scores"])
                correct_labels.extend(x["correct_labels"])
            auc = roc_auc_score(correct_labels, scores)
            # print(seqs[:10])
            thres = [0.5]
            for t in thres:
                print('=' * 50)
                print('The threshold is ', t)
                predict_labels = [1. if i >= t else 0. for i in scores]
                acc = accuracy_score(correct_labels, predict_labels)
                precision = precision_score(correct_labels, predict_labels)
                recall = recall_score(correct_labels, predict_labels)
                f1 = f1_score(correct_labels, predict_labels)
                # self.target_eval(np.array(correct_labels), np.array(predict_labels), np.array(scores), seqs)
                print(
                    " acc: {},".format(acc),
                    " precision: {},".format(precision),
                    " recall: {},".format(recall),
                    " f1_score: {}".format(f1),
                )
            result = {
                "test_loss": avg_loss.item(),
                "auc": auc,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            self.log("final_metric", result)
            # print('final metrics: ', result)

        elif self.args.objective == "regression":
            predict_values, correct_values = [], []
            for x in outputs:
                # print('????', x["correct_values"])
                correct_values.extend(x["correct_values"])
                predict_values.extend(x["predict_values"])
            print(len(predict_values))
            mse = mean_squared_error(correct_values, predict_values)
            rmse = np.sqrt(mean_squared_error(correct_values, predict_values))
            r2 = self.get_rm2(correct_values, predict_values)
            pr = pearson(predict_values, correct_values)
            sr = spearman(predict_values, correct_values)

            # data = pd.read_csv(os.path.join(self.root_data_path, "split_data/test_fold0.csv"))
            # data['pred_label'] = predict_values
            # data.to_csv('evaluate_results.csv', index=False)

            result = {
                "test_loss": avg_loss.item(),
                "mse": mse,
                "rmse": rmse,
                "r2_score": r2,
                "pearson": pr,
                "spearman": sr,
                'ci': ci(correct_values, predict_values)
            }
            # self.log("final_metric", result)
            print(result)
            return result
