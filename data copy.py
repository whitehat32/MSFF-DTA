# -*- encoding: utf-8 -*-
'''
@Time       : 2022/11/26 17:19
@Author     : Tian Tan
@Email      :
@File       : data.py
@Project    :
@Description:
'''
import os

import pandas as pd
import torch
import torch_geometric.utils as utils
from gensim.models import Word2Vec, word2vec
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from tqdm import *

from preprocessing.compound import get_mol_features
from preprocessing.protein import ProteinFeatureManager

import numpy as np


class CustomData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''

    def __inc__(self, key, value, *args, **kwargs):
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        return super().__inc__(key, value, *args, **kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"


class CPIDataset_case(Dataset):
    """CPI数据集

    Args:
        file_path: 数据文件路径（包含化合物SMILES、蛋白质氨基酸序列、标签）
        protein_feature_manager: 蛋白质特征管理器
    """

    def __init__(self, file_path, protein_feature_manager, args, training=False):
        self.args = args
        self.root_data_path = args.root_data_path
        self.raw_data = pd.read_csv(file_path)
        # compound_smiles = pd.read_csv(os.path.join(self.root_data_path, 'compound_smiles.csv'))
        # self.compound_smiles = compound_smiles['smiles']
        self.compound_smiles = self.raw_data['COMPOUND_SMILES'].values
        self.smiles_values = self.raw_data['COMPOUND_SMILES'].values
        self.protein_sequence_values = self.raw_data['PROTEIN_SEQUENCE'].values
        self.protein_id = self.raw_data['PROTEIN_ID'].values

        # if args.objective == 'classification':
            # self.label_values = self.raw_data['CLS_LABEL'].values
        # else:
            # self.label_values = self.raw_data['REG_LABEL'].values

        self.atom_dim = args.atom_dim
        self.protein_feature_manager = protein_feature_manager
        # compound_smiles = pd.read_csv(file_path)['COMPOUND_SMILES']
        # import pdb; pdb.set_trace()
        # compound_smiles = pd.read_csv(os.path.join(self.root_data_path, 'compound_smiles.csv'))
        # self.compound_smiles = compound_smiles['smiles']
        # self.compound_smiles = compound_smiles
        self.compound_fp = dict()

        # random partition
        # self.mol2vec_model = word2vec.Word2Vec.load("mol2vec/model_300dim.pkl")
        self.mol2vec_model = None
        self.compound_graph = dict()
        self.compound_degree = dict()
        self.compound_word_emb = dict()
        self.compound_spatial_pos = dict()
        self.get_mol_graph()
        # self.mol2vec_model = word2vec.Word2Vec.load("..\\mol2vec\\model_300dim.pkl")

    def get_mol_graph(self):
        print("Extracting compound graph")
        for s in tqdm(self.compound_smiles):
            s = Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
            self.compound_graph[s], self.compound_degree[s], compound_fp, compound_spatial_pos = get_mol_features(s, model=self.mol2vec_model, atom_dim=self.atom_dim)
            # self.compound_graph[s] = self.transform_compound(self.compound_graph[s])
            # print(self.compound_graph[s].coarsen_adj)
            # self.compound_spatial_pos[s] = None
            self.compound_spatial_pos[s] = compound_spatial_pos
            self.compound_fp[s] = compound_fp
        print('done')

    def __len__(self):
        return len(self.raw_data)

    def pad_spatial_pos_unsqueeze(self, x, padlen):
        x = x + 1
        xlen = x.size(0)
        if xlen < padlen:
            new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
            new_x[:xlen, :xlen] = x
            x = new_x
        return x.unsqueeze(0)
    def __getitem__(self, idx):
        smiles = self.smiles_values[idx]
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        sequence = self.protein_sequence_values[idx]
        protein_id = self.protein_id[idx]
        # label = self.label_values[idx]
        protein_node_feat = torch.FloatTensor(self.protein_feature_manager.protein_node_feature_dict[protein_id])
        protein_fg = torch.LongTensor(self.protein_feature_manager.protein_fg[protein_id])
        # protein_fg2 = torch.LongTensor(self.protein_feature_manager.protein_fg2[protein_id])
        # protein_fg3 = torch.LongTensor(self.protein_feature_manager.protein_fg3[protein_id])
        # protein_pretrain_emb = self.protein_feature_manager.protein_pretrain_embedding[protein_id]
        # protein_graph = CustomData(x=torch.FloatTensor(protein_node_features),
        #                            edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
        #                            # complete_edge_index=torch.LongTensor(complete_edge_index).transpose(1, 0),
        #                            complete_edge_index=complete_edge_index,
        #                            complete_edge_attr=complete_edge_attr, pretrain_embedding=pretrain_embedding)
        compound_graph = self.compound_graph[smiles]
        compound_degree = self.compound_degree[smiles]
        compound_fp = self.compound_fp[smiles]
        compound_spatial_pos = self.compound_spatial_pos[smiles]
        return {
            'COMPOUND_GRAPH': compound_graph,
            'COMPOUND_FP': compound_fp,
            'COMPOUND_DEGREE': compound_degree,
            'COMPOUND_POS': compound_spatial_pos,
            'PROTEIN_NODE_FEAT': protein_node_feat,
            'PROTEIN_FG': protein_fg,
            # 'PROTEIN_FG2': protein_fg2,
            # 'PROTEIN_FG3': protein_fg3,
            'PROTEIN_PRETRAIN_EMB': None,
            # 'LABEL': label,
            'SEQUENCE': sequence,
        }

    def collate_fn(self, batch):
        """自定义数据合并方法，将数据集中的数据通过Padding构造成相同Size

        Args:
            batch: 原始数据列表

        Returns:
            batch: 经过Padding等处理后的PyTorch Tensor字典
        """
        batch_size = len(batch)

        protein_nums = [item['PROTEIN_NODE_FEAT'].shape[0] for item in batch]
        max_protein_len = max(protein_nums)
        max_protein_fg_len = max_protein_len - 2
        # max_protein_fg_len2 = (max_protein_len - 3) // 2 + 1
        # max_protein_fg_len3 = (max_protein_len - 3) // 3 + 1



        protein_fg = torch.zeros((batch_size, max_protein_fg_len)).long()
        # protein_fg2 = torch.zeros((batch_size, max_protein_fg_len2)).long()
        # protein_fg3 = torch.zeros((batch_size, max_protein_fg_len3)).long()

        protein_nums = torch.LongTensor(protein_nums)
        protein_node_feat = torch.zeros((batch_size, max_protein_len, 21))

        compound_node_nums = [item['COMPOUND_GRAPH'].num_nodes for item in batch]
        max_compound_len = max(compound_node_nums)
        compound_degree = torch.zeros((batch_size, max_compound_len))
        compound_fp = torch.zeros((batch_size, 2727))
        compound_spatial_pos = [self.pad_spatial_pos_unsqueeze(item['COMPOUND_POS'], max_compound_len) for item in batch]
        compound_spatial_pos = torch.stack(compound_spatial_pos, dim=0)
        # compound_spatial_pos = None
        compound_graph = list()
        labels, seqs = list(), list()

        for i, item in enumerate(batch):
            v = item['COMPOUND_GRAPH']
            compound_graph.append(v)

            v = item['COMPOUND_DEGREE']
            compound_degree[i, :v.shape[0]] = torch.FloatTensor(v)

            v = item['COMPOUND_FP']
            compound_fp[i] = torch.LongTensor(v)

            v = item['PROTEIN_NODE_FEAT']
            protein_node_feat[i, :v.shape[0]] = v

            # v = item['PROTEIN_PRETRAIN_EMB']
            # protein_pretrain_emb[i, :v.shape[0]] = torch.FloatTensor(v)

            v = item['PROTEIN_FG']
            protein_fg[i, :v.shape[0]] = torch.LongTensor(v)

            # v = item['PROTEIN_FG2']
            # protein_fg2[i, :v.shape[0]] = torch.LongTensor(v)
            #
            # v = item['PROTEIN_FG3']
            # protein_fg3[i, :v.shape[0]] = torch.LongTensor(v)

            # labels.append(item['LABEL'])
            # seqs.append(item['SEQUENCE'])

        compound_graph = Batch.from_data_list(compound_graph)
        labels = torch.tensor(labels).type(torch.float32)
        return {
            'COMPOUND_GRAPH': compound_graph,
            'COMPOUND_MAX_LEN': max_compound_len,
            'COMPOUND_DEGREE': compound_degree,
            'COMPOUND_FP': compound_fp,
            'COMPOUND_POS': compound_spatial_pos,
            'PROTEIN_NODE_FEAT': protein_node_feat,
            'PROTEIN_PRETRAIN_EMB': None,
            'PROTEIN_NUMS': protein_nums,
            'PROTEIN_FG': protein_fg,
            # 'PROTEIN_FG2': protein_fg2,
            # 'PROTEIN_FG3': protein_fg3,
            # 'LABEL': labels
        }





class CPIDataset(Dataset):
    """CPI数据集

    Args:
        file_path: 数据文件路径（包含化合物SMILES、蛋白质氨基酸序列、标签）
        protein_feature_manager: 蛋白质特征管理器
    """

    def __init__(self, file_path, protein_feature_manager, args, training=False):
        self.args = args
        self.root_data_path = args.root_data_path
        self.raw_data = pd.read_csv(file_path)
        self.smiles_values = self.raw_data['COMPOUND_SMILES'].values
        self.protein_sequence_values = self.raw_data['PROTEIN_SEQUENCE'].values
        self.protein_id = self.raw_data['PROTEIN_ID'].values

        if args.objective == 'classification':
            self.label_values = self.raw_data['CLS_LABEL'].values
        else:
            self.label_values = self.raw_data['REG_LABEL'].values

        self.atom_dim = args.atom_dim
        self.protein_feature_manager = protein_feature_manager

        compound_smiles = pd.read_csv(os.path.join(self.root_data_path, 'compound_smiles.csv'))
        self.compound_smiles = compound_smiles['smiles']
        self.compound_fp = dict()

        # random partition
        # self.mol2vec_model = word2vec.Word2Vec.load("mol2vec/model_300dim.pkl")
        self.mol2vec_model = None
        self.compound_graph = dict()
        self.compound_degree = dict()
        self.compound_word_emb = dict()
        self.compound_spatial_pos = dict()
        self.get_mol_graph()
        # self.mol2vec_model = word2vec.Word2Vec.load("..\\mol2vec\\model_300dim.pkl")

    def get_mol_graph(self):
        print("Extracting compound graph")
        for s in self.compound_smiles:
            s = Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
            self.compound_graph[s], self.compound_degree[s], compound_fp, compound_spatial_pos = get_mol_features(s, model=self.mol2vec_model, atom_dim=self.atom_dim)
            # self.compound_graph[s] = self.transform_compound(self.compound_graph[s])
            # print(self.compound_graph[s].coarsen_adj)
            # self.compound_spatial_pos[s] = None
            self.compound_spatial_pos[s] = compound_spatial_pos
            self.compound_fp[s] = compound_fp
        print('done')

    def __len__(self):
        return len(self.raw_data)

    def pad_spatial_pos_unsqueeze(self, x, padlen):
        x = x + 1
        xlen = x.size(0)
        if xlen < padlen:
            new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
            new_x[:xlen, :xlen] = x
            x = new_x
        return x.unsqueeze(0)
    def __getitem__(self, idx):
        smiles = self.smiles_values[idx]
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        sequence = self.protein_sequence_values[idx]
        protein_id = self.protein_id[idx]
        label = self.label_values[idx]
        protein_node_feat = torch.FloatTensor(self.protein_feature_manager.protein_node_feature_dict[protein_id])
        protein_fg = torch.LongTensor(self.protein_feature_manager.protein_fg[protein_id])
        # protein_fg2 = torch.LongTensor(self.protein_feature_manager.protein_fg2[protein_id])
        # protein_fg3 = torch.LongTensor(self.protein_feature_manager.protein_fg3[protein_id])
        # protein_pretrain_emb = self.protein_feature_manager.protein_pretrain_embedding[protein_id]
        # protein_graph = CustomData(x=torch.FloatTensor(protein_node_features),
        #                            edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
        #                            # complete_edge_index=torch.LongTensor(complete_edge_index).transpose(1, 0),
        #                            complete_edge_index=complete_edge_index,
        #                            complete_edge_attr=complete_edge_attr, pretrain_embedding=pretrain_embedding)
        compound_graph = self.compound_graph[smiles]
        compound_degree = self.compound_degree[smiles]
        compound_fp = self.compound_fp[smiles]
        compound_spatial_pos = self.compound_spatial_pos[smiles]
        return {
            'COMPOUND_GRAPH': compound_graph,
            'COMPOUND_FP': compound_fp,
            'COMPOUND_DEGREE': compound_degree,
            'COMPOUND_POS': compound_spatial_pos,
            'PROTEIN_NODE_FEAT': protein_node_feat,
            'PROTEIN_FG': protein_fg,
            # 'PROTEIN_FG2': protein_fg2,
            # 'PROTEIN_FG3': protein_fg3,
            'PROTEIN_PRETRAIN_EMB': None,
            'LABEL': label,
            'SEQUENCE': sequence,
        }

    def collate_fn(self, batch):
        """自定义数据合并方法，将数据集中的数据通过Padding构造成相同Size

        Args:
            batch: 原始数据列表

        Returns:
            batch: 经过Padding等处理后的PyTorch Tensor字典
        """
        batch_size = len(batch)

        protein_nums = [item['PROTEIN_NODE_FEAT'].shape[0] for item in batch]
        max_protein_len = max(protein_nums)
        max_protein_fg_len = max_protein_len - 2
        # max_protein_fg_len2 = (max_protein_len - 3) // 2 + 1
        # max_protein_fg_len3 = (max_protein_len - 3) // 3 + 1



        protein_fg = torch.zeros((batch_size, max_protein_fg_len)).long()
        # protein_fg2 = torch.zeros((batch_size, max_protein_fg_len2)).long()
        # protein_fg3 = torch.zeros((batch_size, max_protein_fg_len3)).long()

        protein_nums = torch.LongTensor(protein_nums)
        protein_node_feat = torch.zeros((batch_size, max_protein_len, 21))

        compound_node_nums = [item['COMPOUND_GRAPH'].num_nodes for item in batch]
        max_compound_len = max(compound_node_nums)
        compound_degree = torch.zeros((batch_size, max_compound_len))
        compound_fp = torch.zeros((batch_size, 2727))
        compound_spatial_pos = [self.pad_spatial_pos_unsqueeze(item['COMPOUND_POS'], max_compound_len) for item in batch]
        compound_spatial_pos = torch.stack(compound_spatial_pos, dim=0)
        # compound_spatial_pos = None
        compound_graph = list()
        labels, seqs = list(), list()

        for i, item in enumerate(batch):
            v = item['COMPOUND_GRAPH']
            compound_graph.append(v)

            v = item['COMPOUND_DEGREE']
            compound_degree[i, :v.shape[0]] = torch.FloatTensor(v)

            v = item['COMPOUND_FP']
            compound_fp[i] = torch.LongTensor(v)

            v = item['PROTEIN_NODE_FEAT']
            protein_node_feat[i, :v.shape[0]] = v

            # v = item['PROTEIN_PRETRAIN_EMB']
            # protein_pretrain_emb[i, :v.shape[0]] = torch.FloatTensor(v)

            v = item['PROTEIN_FG']
            protein_fg[i, :v.shape[0]] = torch.LongTensor(v)

            # v = item['PROTEIN_FG2']
            # protein_fg2[i, :v.shape[0]] = torch.LongTensor(v)
            #
            # v = item['PROTEIN_FG3']
            # protein_fg3[i, :v.shape[0]] = torch.LongTensor(v)

            labels.append(item['LABEL'])
            # seqs.append(item['SEQUENCE'])

        compound_graph = Batch.from_data_list(compound_graph)
        labels = torch.tensor(labels).type(torch.float32)
        return {
            'COMPOUND_GRAPH': compound_graph,
            'COMPOUND_MAX_LEN': max_compound_len,
            'COMPOUND_DEGREE': compound_degree,
            'COMPOUND_FP': compound_fp,
            'COMPOUND_POS': compound_spatial_pos,
            'PROTEIN_NODE_FEAT': protein_node_feat,
            'PROTEIN_PRETRAIN_EMB': None,
            'PROTEIN_NUMS': protein_nums,
            'PROTEIN_FG': protein_fg,
            # 'PROTEIN_FG2': protein_fg2,
            # 'PROTEIN_FG3': protein_fg3,
            'LABEL': labels
        }


if __name__ == "__main__":
    galaxydb_data_path = '../data/Davis/'
    protein_feature_manager = ProteinFeatureManager(galaxydb_data_path)

    args = main.parse_args()
    train_set = CPIDataset(galaxydb_data_path + 'prot_cold_start/train_fold0.csv', protein_feature_manager, args)

    # # valid_set = CPIDataset(galaxydb_data_path + 'split_data/valid.csv', protein_feature_manager)
    # test_set = CPIDataset(galaxydb_data_path + 'split_data_v1/prot_cold_start/test_fold0.csv', protein_feature_manager, args)
    # raw_train_data = pd.read_csv(galaxydb_data_path + 'split_data_v1/prot_cold_start/test_fold0.csv')
    # raw_train_data = pd.read_csv(galaxydb_data_path + 'split_data_v1/prot_cold_start/test_fold0.csv')
    # cp_smiles = raw_train_data['COMPOUND_SMILES'].values
    # avg_num = 0
    # for s in cp_smiles:
    #     try:
    #         mol = Chem.MolFromSmiles(s)
    #     except Exception:
    #         raise RuntimeError("SMILES cannot been parsed!")
    #     avg_num += mol.GetNumAtoms()
    # print('avg_num', avg_num / len(cp_smiles))

# print('Test Item:')
# print('Compound Node Feature Shape:', item['COMPOUND_NODE_FEAT'].shape)
# # print('Compound Adjacency Matrix Shape:', item['COMPOUND_ADJ'].shape)
# print('Protein Node Feature Shape:', item['PROTEIN_NODE_FEAT'].shape)
# print('Protein Contact Map Shape:', item['PROTEIN_MAP'].shape)
# print('Protein Embedding Shape:', item['PROTEIN_EMBEDDING'].shape)
# print('Label:', item['LABEL'])

#     train_loader = DataLoader(
#         train_set,
#         batch_size=32,
#         collate_fn=train_set.collate_fn,
#         shuffle=False,
#         num_workers=1,
#         drop_last=True,
#     )
# #
# # print('')
# # print('Test Batch:')
# # avg_node_num = 0
#     for batch in train_loader:
#         print('Compound Node Feature Shape:', batch['PROTEIN_GRAPH'])

# print('Compound Adjacency Matrix Shape:', batch['COMPOUND_ADJ'].shape)
# print('Compound Node Numbers Shape:', batch['COMPOUND_NODE_NUM'].shape)
# print('Protein Node Feature Shape:', batch['PROTEIN_NODE_FEAT'].shape)
# print('Protein Contact Map Shape:', batch['PROTEIN_MAP'].shape)
# print('Protein Embedding Shape:', batch['PROTEIN_EMBEDDING'].shape)
# print('Protein Node Numbers Shape:', batch['PROTEIN_NODE_NUM'].shape)
# print('Label Shape:', batch['LABEL'].shape)


