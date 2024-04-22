# -*- encoding: utf-8 -*-


import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from model.gnn import MultiGCN, MultiGCN_cp, ResGCN, GIN
from torch_geometric.nn import GraphConv, global_max_pool, global_add_pool, global_mean_pool, InstanceNorm
from torch_geometric.utils import to_dense_batch, softmax, sort_edge_index, scatter
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK
# from torch_geometric.nn.pool.topk_pool import filter_adj
from torch_scatter import scatter_add, scatter_max, scatter_mean
from torch_sparse import transpose
from torch_geometric.utils.num_nodes import maybe_num_nodes



class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, seq_pos=None):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        self.seq_pos = seq_pos
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim // n_heads)
        self.cnt = 0

    def forward(self, query, key, value, mask=None, compound_pos=None):
        # query = key = value [batch size, sent_len, hid_dim]
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)


        # Q, K, V = [batch_size, sent_len, hid_dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch_size, n_heads, sent_len_K, hid_dim / n_heads]
        # Q = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, sent_len_Q, sent_len_K]

        if compound_pos is not None:
            # energy += self.seq_pos(Q.shape[2], K.shape[2])
            energy += compound_pos

        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
            attention = (F.softmax(energy, dim=-1))
        else:
            attention = (F.softmax(energy, dim=-1))
        

        
        attention = self.do(attention)

        x = torch.matmul(attention, V)
        # x = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, sent_len_Q, n_heads, hid_dim / n_heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch_size, sent_len_Q, hid_dim]
        x = self.fc(x)
        # x = [batch_size, sent_len_Q, hid_dim]

        return x


class SelfAttention_sig(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.w_q1 = nn.Linear(hid_dim, hid_dim)
        self.w_k1 = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim // n_heads)

    def softmax(self, input, sample=None,epsilon=1e-6):
        x = input.exp()
        if sample is not None:
            x = x * sample
        partition = x.sum(dim=-1, keepdim=True)
        return x / (partition+epsilon)

    def forward(self, query, key, value, mask=None):
        # query = key = value [batch size, sent_len, hid_dim]
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q1 = self.w_q1(query)
        K1 = self.w_k1(key)

        # Q, K, V = [batch_size, sent_len, hid_dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)


        Q1 = Q1.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K1 = K1.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch_size, n_heads, sent_len_K, hid_dim / n_heads]
        # Q = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, sent_len_Q, sent_len_K]
        sample = torch.matmul(Q1, K1.permute(0, 1, 3, 2)) / self.scale
        sample = concrete_sample(sample, 0.2, self.training)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
            attention = F.softmax(energy, dim=-1) 
            attention = attention * sample
            # attention = F.softmax(energy, dim=-1)
            # attention = attention * sample

        r_att = torch.mean(attention, dim=1)

        # attention = self.do(attention)
        x = torch.matmul(attention, V)
        # x = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, sent_len_Q, n_heads, hid_dim / n_heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch_size, sent_len_Q, hid_dim]
        x = self.fc(x)
        # x = [batch_size, sent_len_Q, hid_dim]

        return x, r_att

class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=1500,  n_heads=1):
        super(RelativePositionBias, self).__init__()
        self.num_buckets = num_buckets
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)
        self.relative_attention_bias.weight = torch.nn.Parameter(torch.ones_like(self.relative_attention_bias.weight))

    @staticmethod
    def _relative_position_bucket(relative_position):
        """
        Args:
            relative_position: an int32 Tensor

        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """

        n = relative_position
        maxn = torch.max(n)
        n = torch.where(n < 0, -n + maxn, n)

        return n

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        rp_bucket = self._relative_position_bucket(
            relative_position  # shape (qlen, klen)
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, qlen, klen):
        return self.compute_bias(qlen, klen)  # shape (1, num_heads, qlen, klen)

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hid_dim)
        self.ff_linear1 = nn.Linear(hid_dim, hid_dim * 2)
        self.ff_linear2 = nn.Linear(hid_dim * 2, hid_dim)
        self.do = nn.Dropout(dropout)
        self.protein_gcn = MultiGCN(hid_dim, hid_dim)
        self.act = F.relu

    def forward(self, src, src_graph, src_mask=None):
        local_src = self.protein_gcn(src, src_graph)

        global_g_src = src + self.do(self.sa(local_src, local_src, src, src_mask))
        # 拼接
        global_src = global_g_src
        output = self.norm(src + self._ff_block(global_src))

        return output

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.do(self.act(self.ff_linear1(x)))
        return self.do(self.ff_linear2(x))


class Encoder(nn.Module):
    """protein feature extraction."""

    def __init__(self, protein_dim, hid_dim, n_layers, dropout, n_heads):
        super().__init__()

        # assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, hid_dim], requires_grad=True))

    def forward(self, protein, protein_graph, protein_mask, train=1):
        for i, layer in enumerate(self.layers):
            protein = layer(protein, protein_graph, protein_mask)
        output = protein
        return output

    def concrete_sample(self, att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, sent_len, hid_dim]

        x = x.permute(0, 2, 1)
        # x = [batch_size, hid_dim, sent_len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch_size, pf_dim, sent_len]
        x = self.fc_2(x)
        # x = [batch_size, hid_dim, sent_len]
        x = x.permute(0, 2, 1)
        # x = [batch_size, sent_len, hid_dim]
        return x


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch=None):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs)
            else:
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


class StructureExtractor(nn.Module):
    r""" K-subtree structure extractor. Computes the structure-aware node embeddings using the
    k-hop subtree centered around each node.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree
    """

    def __init__(self, hid_dim, gnn_type="gcn", num_layers=3,
                 batch_norm=False, concat=True, khopgnn=False, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gcn = GIN(hid_dim, dropout, num_layers)
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * hid_dim if concat else hid_dim

        self.out_proj = nn.Linear(inner_dim, hid_dim)

    def forward(self, x, edge_index, subgraph_indicator_index=None, agg="sum", edge_attr=None):
        x_cat = [x]
        x = self.gcn(x, edge_index)
        x_cat.extend(x)
        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        if self.khopgnn:
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            return x

        if self.num_layers > 0 and self.batch_norm:
            # x = self.bn(x)
            pass
        x = self.out_proj(x)
        return x


class KHopStructureExtractor(nn.Module):
    r""" K-subgraph structure extractor. Extracts a k-hop subgraph centered around
    each node and uses a GNN on each subgraph to compute updated structure-aware
    embeddings.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree (True)
    """

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3,
                 khopgnn=True):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn

        self.structure_extractor = StructureExtractor(
            embed_dim,
            gnn_type=gnn_type,
            num_layers=num_layers,
            concat=False,
            khopgnn=True
        )
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x, subgraph_edge_index, subgraph_indicator_index=None,
                subgraph_node_index=None, subgraph_edge_attr=None):
        # print('sub = ', subgraph_edge_index)
        x_struct = self.structure_extractor(
            x[subgraph_node_index],
            subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            subgraph_indicator_index=subgraph_indicator_index,
            agg="sum",
        )
        x_struct = torch.cat([x, x_struct], dim=-1)
        x_struct = self.out_proj(x_struct)

        return x_struct


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

class SelfAttBlock(nn.Module):

    def __init__(self, dim, channel_dim, dropout=0., seq_pos=None):
        super().__init__()
        self.seq_pos = seq_pos
        self.ln = nn.LayerNorm(dim)
        self.do = nn.Dropout(dropout)
        self.att_blk = SelfAttention(dim, 1, dropout, seq_pos=self.seq_pos)
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x, trg_mask, compound_pos):
        x = self.ln(x)
        x = x + self.do(self.ln(self.att_blk(x, x, x, trg_mask.unsqueeze(1).unsqueeze(2), compound_pos=compound_pos)))
        x = x + self.channel_mix(x)

        return x





class MSGT(nn.Module):
    r"""
    Args:
    ----------
    embed_dim (int):        the embeding dimension
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree
    """

    def __init__(self, hid_dim, num_layers=3, concat=True, khopgnn=True, dropout=0.2, seq_pos=None):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gin = GIN(hid_dim, dropout, num_layers)
        
        self.hid_dim = hid_dim
        self.mlp_list = nn.ModuleList()
        for i in range(3):
            self.mlp_list.append(SelfAttBlock(dim=hid_dim, channel_dim=hid_dim, dropout=dropout, seq_pos=RelativePositionBias()))

        self.lin1 = FeedForward(hid_dim * 2, hid_dim, dropout=dropout)
        self.lin2 = FeedForward(hid_dim * 2, hid_dim, dropout=dropout)
        self.lin3 = FeedForward(hid_dim * 2, hid_dim, dropout=dropout)
        self.sub_lin1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hid_dim, hid_dim*2), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(hid_dim*2, hid_dim)
                        )
        self.sub_lin2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hid_dim, hid_dim*2), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(hid_dim*2, hid_dim)
                        )
        self.sub_lin3 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hid_dim, hid_dim*2), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(hid_dim*2, hid_dim)
                        )
        
        self.node_lin1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hid_dim * 2, hid_dim), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(hid_dim, hid_dim)
                        )
        self.node_lin2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hid_dim * 2, hid_dim), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(hid_dim, hid_dim)
                        )
        self.node_lin3 = nn.Sequential(nn.Dropout(dropout), nn.Linear(hid_dim * 2, hid_dim), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(hid_dim, hid_dim)
                        )
        self.mfb1 = MFB(hid_dim, hid_dim, hid_dim*2)
        self.mfb2 = MFB(hid_dim, hid_dim, hid_dim*2)
        self.mfb3 = MFB(hid_dim, hid_dim, hid_dim*2)
        self.node_mfb1 = MFB(hid_dim, hid_dim, hid_dim*2)
        self.node_mfb2 = MFB(hid_dim, hid_dim, hid_dim*2)
        self.node_mfb3 = MFB(hid_dim, hid_dim, hid_dim*2)

    # 由gnn选择主骨架
    def forward(self, x, edge_index, compound_batch=None, subgraph_indicator_index=None, edge_index_2hop=None,
                edge_index_3hop=None,
                agg="sum", edge_attr=None, edge_atten=None, compound_pos=None):
        x_out_local = []
        x_out_local_node = []
        x_out_global = []
        x_out_global_node = []
        x_list = self.gin(x, edge_index, concat=True, edge_atten=edge_atten)
        x_1hop = scatter(x_list[0][edge_index[0]], edge_index[1], dim=0, dim_size=x.size(0), reduce=agg)
        x_out_local_node.append(self.sub_lin1(x_1hop))
        x_out_local.append(global_mean_pool(x_1hop, compound_batch))
        if edge_index_2hop is not None:
            x_2hop = scatter(x_list[1][edge_index_2hop[0]], edge_index_2hop[1], dim=0, dim_size=x.size(0),
                            reduce=agg)
        else:
            x_2hop = x_list[1]
        x_out_local_node.append(self.sub_lin2(x_2hop))
        x_out_local.append(global_mean_pool(x_2hop, compound_batch))
        if edge_index_3hop is not None:
            x_3hop = scatter(x_list[2][edge_index_3hop[0]], edge_index_3hop[1], dim=0, dim_size=x.size(0),
                            reduce=agg)
        else:
            x_3hop = x_list[2]
        x_out_local_node.append(self.sub_lin3(x_3hop))
        x_out_local.append(global_mean_pool(x_3hop, compound_batch))
        
        for j in range(3):
            x_global, x_mask = to_dense_batch(x_out_local_node[j], compound_batch)
            for i in range(1):
                x_global = self.mlp_list[j](x_global, x_mask, compound_pos)
            x_out_global_node.append(x_global[x_mask])
            x_out_global.append(global_mean_pool(x_out_global_node[-1], compound_batch))
                
        x1 = self.lin1(self.mfb1(x_out_local[0], x_out_global[0]).squeeze())
        x2 = self.lin2(self.mfb2(x_out_local[1], x_out_global[1]).squeeze())
        x3 = self.lin2(self.mfb3(x_out_local[2], x_out_global[2]).squeeze())

        x1_node = self.node_lin1(self.node_mfb1(x_out_local_node[0], x_out_global_node[0]).squeeze())
        x2_node = self.node_lin2(self.node_mfb2(x_out_local_node[1], x_out_global_node[1]).squeeze())
        x3_node = self.node_lin3(self.node_mfb3(x_out_local_node[2], x_out_global_node[2]).squeeze())
        
        x = torch.cat((x1, x2, x3), dim=-1)
        x_node = torch.cat([x1_node, x2_node, x3_node], dim=-1)
        return x, x_node

vector_operations = {
    "cat": (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim),
    "add": (torch.add, lambda dim: dim),
    "sub": (torch.sub, lambda dim: dim),
    "mul": (torch.mul, lambda dim: dim),
    "combination1": (lambda x, y: torch.cat((x, y, torch.add(x, y)), -1), lambda dim: 3 * dim),
    "combination2": (lambda x, y: torch.cat((x, y, torch.sub(x, y)), -1), lambda dim: 3 * dim),
    "combination3": (lambda x, y: torch.cat((x, y, torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination4": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 2 * dim),
    "combination5": (lambda x, y: torch.cat((torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination6": (lambda x, y: torch.cat((torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination7": (
        lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination8": (lambda x, y: torch.cat((x, y, torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination9": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination10": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 4 * dim),
    "combination11": (
        lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 5 * dim)
}


class LinearBlock(torch.nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = [x]
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)
        return embeddings


class MLP_Decoder(torch.nn.Module):
    def __init__(self, embedding_dim=128, out_dim=1, prediction_mode="cat"):
        super(MLP_Decoder, self).__init__()
        mlp_layers_dim = [embedding_dim, 512, 256, out_dim]
        self.mlp = LinearBlock(mlp_layers_dim, 0.2, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, inputs):
        mlp_embeddings = self.mlp(inputs)
        out = mlp_embeddings[-1]
        return out


class MFB(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, k=4, dropout_rate=0.):
        super(MFB, self).__init__()
        self.hid_dim = hid_dim
        self.k = k
        self.proj_i = nn.Linear(v_dim, k * hid_dim)
        self.proj_q = nn.Linear(q_dim, k * hid_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AvgPool1d(k, stride=k)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, img_feat, ques_feat, exp_in=1, ln=True):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)  # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)  # (N, 1, K*O)

        exp_out = img_feat * ques_feat   # (N, C, K*O)
        exp_out = self.dropout(exp_out)  # (N, C, K*O)
        if ln:
            z = self.pool(exp_out) * self.k  # (N, C, O)
            z = self.ln(z)
        else:
            z = self.pool(exp_out)
        return z



class MFB_FUS(nn.Module):
    def __init__(self, src_dim, trg_dim, hid_dim, k=3, dropout_rate=0.):
        super(MFB_FUS, self).__init__()
        self.hid_dim = hid_dim
        self.k = k
        self.proj_src = nn.Linear(src_dim, k * hid_dim)
        self.self_att = SelfAttention_sig(src_dim, n_heads=3, dropout=dropout_rate)
        self.protein_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = FeedForward(src_dim, k * hid_dim, dropout=dropout_rate)
        self.act = torch.nn.GELU()
        self.pool = nn.AvgPool1d(k, stride=k)
        self.ln = nn.LayerNorm(hid_dim)
        # self.ln1 = nn.LayerNorm(hid_dim * 3)

    def forward(self, src, trg, compound_graph, trg_mask=None, src_num=None, training=True):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = src.shape[0]
        inter_att, r_att = self.self_att(src, trg, trg, trg_mask.unsqueeze(1).unsqueeze(2))
        src = src + self.fc(inter_att)
        
        src = self.protein_pool(src.permute(0, 2, 1)).squeeze(2)

        src = self.proj_src(src)
        exp_out = self.dropout(src)  # (N, C, K*O)
        z = self.pool(exp_out) * self.k  # (N, C, O)
        
        z = z.view(batch_size, self.hid_dim)  # (N, C, O)

        return z, r_att


class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, dropout=0.2, learn_edge_att=False):
        super().__init__()
        self.learn_edge_att = learn_edge_att
        dropout_p = dropout

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12)
        else:
            att_log_logits = self.feature_extractor(emb)
        return att_log_logits


def concrete_sample(att_log_logit, temp, training):
    # random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
    # random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
    # att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
    # print('training:', training)
    if training:
        random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        # output = (att_log_logit + random_noise) / temp
        output = ((att_log_logit + random_noise) / temp).sigmoid()
    else:
        # output = att_log_logit
        output = (att_log_logit).sigmoid()
    return output
          

def lift_node_att_to_edge_att(node_att, edge_index):
    src_lifted_att = node_att[edge_index[0]]
    dst_lifted_att = node_att[edge_index[1]]
    edge_att = src_lifted_att * dst_lifted_att
    return edge_att





def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0].index_select(0, batch) - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if isinstance(ratio, int):
            k = num_nodes.new_full((num_nodes.size(0), ), ratio)
            k = torch.min(k, num_nodes)
        else:
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)
        f_mask = torch.ones_like(perm)
        f_mask[mask] = False
        perm = perm[mask]
        f_perm = perm[f_mask]

    return perm, f_perm



class DPI(nn.Module):
    """ compound feature extraction."""

    def __init__(self, atom_dim, fp_dim, hid_dim, drug_n_layers, drug_n_heads, pf_dim, dropout, objective,
                 tape_emb_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.drug_n_layers = drug_n_layers
        self.drug_n_heads = drug_n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.MGT = MSGT(hid_dim, dropout=dropout, num_layers=3)
        self.sub_gnn3 = GIN(hid_dim, dropout, 3)

        self.ft = nn.Linear(atom_dim, hid_dim)
        self.ft_sub = nn.Linear(atom_dim, hid_dim)
        self.h_mat = nn.Parameter(torch.Tensor(1, 2, 1, hid_dim*3).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, 2, 1, 1).normal_())

        self.do = nn.Dropout(0.5)
        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.5)
        self.do3 = nn.Dropout(0.5)

        self.fp_fc1 = torch.nn.Linear(fp_dim, hid_dim * 6)
        self.src_fc = torch.nn.Linear(hid_dim, hid_dim)

        self.compound_fusion = MFB(hid_dim * 6, hid_dim * 6, hid_dim * 2, dropout_rate=0.1, k=4)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.Tanh()
        self.extractor3 = ExtractorMLP(hid_dim * 3, dropout=self.dropout)
        self.protein_pool = nn.AdaptiveMaxPool1d(1)
        self.protein_node_readout = nn.Linear(hid_dim * 3, hid_dim)
        self.fusion_mfb = MFB_FUS(hid_dim * 3, hid_dim * 3, hid_dim, k=4,dropout_rate=self.dropout)
        self.pridict_mfb = MFB(hid_dim, hid_dim * 2, hid_dim * 2, k=3)
        self.concat_fc = nn.Linear(hid_dim * 3, hid_dim*3)
        self.objective = objective

        if objective == 'classification':
            self.mlp = MLP_Decoder(hid_dim * 2, 2)
        else:
            # 新子图 cat 旧子图
            self.mlp = MLP_Decoder(hid_dim * 2, 1)

    def filter_adj(
        self,
        edge_index,
        edge_attr,
        node_index,
        cluster_index = None,
        num_nodes = None,
    ):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if cluster_index is None:
            cluster_index = torch.arange(node_index.size(0),
                                        device=node_index.device)

        mask = node_index.new_full((num_nodes, ), -1)
        mask[node_index] = cluster_index

        row, col = edge_index[0], edge_index[1]
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

        return torch.stack([row, col], dim=0), edge_attr


    def forward(self, trg, compound_fp, src_sub, compound_graph, protein_num=None, compound_pos=None):

        # trg = [batch_size, compound_len, atom_dim]
        # src = [batch_size, protein_len, hid_dim] # encoder output
        # type= 1 表示训练模式， type = 0表示评估状态

        trg_input = trg.clone()
        trg = self.ft(trg_input)
        trg_sub = self.ft_sub(trg_input)


        trg, trg_list = self.MGT(trg, compound_graph.edge_index, compound_batch=compound_graph.batch,
                                        edge_index_2hop=compound_graph.edge_index_2hop,
                                        edge_index_3hop=compound_graph.edge_index_3hop, compound_pos=compound_pos)
        
        compound_fp1 = self.fp_fc1(compound_fp)
        trg = self.compound_fusion(trg, compound_fp1)
        trg = trg.squeeze()

        sub_node_att3 = trg_list
        sub_node_att3 = self.extractor3(sub_node_att3, compound_graph.edge_index)
        
        sub_node_att3 = concrete_sample(sub_node_att3, 1, training=0).squeeze()
        perm, perm1 = topk(sub_node_att3, 0.5, compound_graph.batch)
        torch.set_printoptions(profile="full")

        batch = compound_graph.batch[perm]
        filter_edge_index, _ = self.filter_adj(compound_graph.edge_index, None, perm, num_nodes=sub_node_att3.size(0))

        sub_graph3 = torch.cat(self.sub_gnn3(trg_sub[perm], filter_edge_index, concat=True), dim=-1)
        sub_graph, sub_mask = to_dense_batch(sub_graph3, batch)
        

        # attention 模块
        inter, r_att = self.fusion_mfb(src_sub, sub_graph, compound_graph, sub_mask, src_num=protein_num, training=self.training)

        outputs = self.mlp(self.pridict_mfb(inter, trg))

        if self.objective == 'regression':
            outputs = outputs.view(-1)
            # outputs = torch.sigmoid(outputs) * 14
        return outputs, r_att, None

    def get_param(self, *shape):
        param = nn.Parameter(torch.zeros(shape))
        nn.init.xavier_normal_(param)
        return param

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]
    


