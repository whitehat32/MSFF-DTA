# -*- encoding: utf-8 -*-
'''
@Time       : 2022/11/26 17:19
@Author     : Tian Tan
@Email      :
@File       : protein.py
@Project    :
@Description:
'''
import os
import pickle

import networkx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx




def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class ProteinFeatureManager():
    def __init__(self, data_path):
        self.pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                              'W', 'Y', 'X']    #'U', 'B', 'O']
        map_data = pd.read_csv(os.path.join(data_path, 'protein_mapping.csv'))
        self.id_to_sequence = {}
        self.fg_dict = {}
        self.protein_fg = {}
        self.protein_fg2 = {}
        self.protein_fg3 = {}
        self.protein_node_feature_dict = {}
        self.protein_contact_graph_dict = {}
        # self.protein_pretrain_embedding = {}
        self.data_path = data_path
        # pretrain_embedding = np.load(os.path.join(data_path, 'pretrain_proteins.npz'), allow_pickle=True)['dict'][()]
        for seq, prot_id in zip(list(map_data['sequences']), list(map_data['prot_id'])):
            self.id_to_sequence[prot_id] = seq
            # seq = seq[:1200]
            category_seq = self.protein2category(seq)
            self.get_fg_dict(category_seq)
            self.protein_fg[prot_id] = self.get_fg(category_seq)
            self.protein_fg2[prot_id] = self.get_fg(category_seq, padding=2)
            self.protein_fg3[prot_id] = self.get_fg(category_seq, padding=3)
            protein_node_feature = self.seq_feature(prot_id)
            self.protein_node_feature_dict[prot_id] = protein_node_feature
            # self.protein_pretrain_embedding[prot_id] = pretrain_embedding[prot_id][0][:][1:-1]

            # self.protein_contact_graph_dict[prot_id] = torch.ones((2, len(seq))).long()
        # self.protein_graph_dict = self.get_protein_graph()

    # def get_protein_graph(self, training=False):
    #     protein_graph_dict = {}
    #     for prot_id in self.protein_node_features_dict:
    #         # protein_graph = Data(x=torch.FloatTensor(self.protein_node_features_dict[prot_id]),
    #         #                      edge_index=torch.LongTensor(self.protein_contact_graph_dict[prot_id]).transpose(1, 0))
    #
    #         protein_graph = Data(x=torch.FloatTensor(self.protein_node_features_dict[prot_id]))
    #
    #         if not training:
    #             protein_graph_dict[prot_id] = protein_graph
    #             # protein_graph_dict[prot_id] = self.transform_protein_test(protein_graph)
    #         else:
    #             protein_graph_dict[prot_id] = self.transform_protein_train(protein_graph)
    #
    #     return protein_graph_dict

    def protein2category(self, p):
        dict = {}
        dict['H'] = 'A'
        dict['R'] = 'A'
        dict['K'] = 'A'

        dict['D'] = 'B'
        dict['E'] = 'B'
        dict['N'] = 'B'
        dict['Q'] = 'B'
        dict['B'] = 'B'

        dict['C'] = 'C'
        dict['X'] = 'C'

        dict['S'] = 'D'
        dict['T'] = 'D'
        dict['P'] = 'D'
        dict['A'] = 'D'
        dict['G'] = 'D'
        dict['U'] = 'D'
        dict['O'] = 'D'

        dict['M'] = 'E'
        dict['I'] = 'E'
        dict['L'] = 'E'
        dict['V'] = 'E'

        dict['F'] = 'F'
        dict['Y'] = 'F'
        dict['W'] = 'F'
        tmp = ''
        for j, amino in enumerate(p):
            try:
                tmp = tmp + dict[amino]
            except:
                tmp = tmp + 'G'
                print('bad amino', amino)
                # num=num+
        return tmp

    def get_fg_dict(self, seq, ngram=3):

        for i in range(len(seq) - ngram + 1):
            fg_str = seq[i: i + ngram]
            if fg_str not in self.fg_dict:
                self.fg_dict[fg_str] = len(self.fg_dict) + 1
        return self.fg_dict

    def get_fg(self, seq, ngram=3, padding=1):
        protein_fg = []
        for i in range(0, len(seq) - ngram + 1, padding):
            fg_str = seq[i: i + ngram]
            protein_fg.append(self.fg_dict[fg_str])
        return np.array(protein_fg)

    def dic_normalize(self, dic):
        # print(dic)
        max_value = dic[max(dic, key=dic.get)]
        min_value = dic[min(dic, key=dic.get)]
        # print(max_value)
        interval = float(max_value) - float(min_value)
        for key in dic.keys():
            dic[key] = (dic[key] - min_value) / interval
        dic['X'] = (max_value + min_value) / 2.0
        return dic

    def get_node_features(self, prot_id):
        sequence = self.id_to_sequence[prot_id]
        return np.load(os.path.join(self.data_path, 'protein_node_features', self.id_to_sequence[sequence] + '.npy'))

    def get_contact_map(self, prot_id):
        pt_map = np.load(os.path.join(self.data_path, 'pconsc4', prot_id + '.npy'))
        sequence = self.id_to_sequence[prot_id]
        pt_map = pt_map[:len(sequence), :len(sequence)]
        pt_map += np.matrix(np.eye(len(sequence)))
        # pt_map = (pt_map >= 0.5).astype(int)
        index_row, index_col = np.where(pt_map >= 0.5)
        target_edge_index = []
        for i, j in zip(index_row, index_col):
            target_edge_index.append([i, j])

        target_edge_index = np.array(target_edge_index)
        return target_edge_index

        # 临时改为输出pyg图

        # target_graph = Data(edge_index=torch.LongTensor(target_edge_index).transpose(1, 0), num_nodes=len(sequence))
        # target_graph = to_networkx(target_graph)
        # return target_graph

        # return np.where(pt_map > 0.1, pt_map, 0)

    def get_local_att_edge(self, n, d=7):
        s1 = torch.zeros((2 * d + 1) * n - d * d - d).long()
        s2 = torch.zeros((2 * d + 1) * n - d * d - d).long()

        protein_local_att_edge_type = torch.zeros((2 * d + 1) * n - d * d - d).long()
        local_att = 2 * d + 1
        tmp = 0
        for i in range(n):
            for j in range(max(0, i - d), min(n - 1, i + d) + 1):
                s1[tmp] = j
                s2[tmp] = i
                protein_local_att_edge_type[tmp] = (i - j + local_att) % local_att
                tmp += 1
        protein_local_att_edge_attr = torch.nn.functional.one_hot(protein_local_att_edge_type, num_classes=2 * d + 1)
        protein_local_att_edge_attr = torch.cat([protein_local_att_edge_attr
                                                    , protein_local_att_edge_type.float().unsqueeze(1)], dim=-1)
        # print(protein_local_att_edge_attr.shape)
        protein_local_att_graph_index = torch.vstack((s1, s2))
        return protein_local_att_graph_index, protein_local_att_edge_attr

    def residue_features(self, residue):
        res_property1 = [1 if residue in self.pro_res_aliphatic_table else 0,
                         1 if residue in self.pro_res_aromatic_table else 0,
                         1 if residue in self.pro_res_polar_neutral_table else 0,
                         1 if residue in self.pro_res_acidic_charged_table else 0,
                         1 if residue in self.pro_res_basic_charged_table else 0]
        res_property2 = [self.res_weight_table[residue], self.res_pka_table[residue], self.res_pkb_table[residue],
                         self.res_pkx_table[residue],
                         self.res_pl_table[residue], self.res_hydrophobic_ph2_table[residue],
                         self.res_hydrophobic_ph7_table[residue]]
        # print(np.array(res_property1 + res_property2).shape)
        return np.array(res_property1 + res_property2)

    # one ont encoding
    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def seq_feature(self, prot_id):
        pro_seq = self.id_to_sequence[prot_id]
        pro_hot = np.zeros((len(pro_seq), len(self.pro_res_table)))
        # pro_property = np.zeros((len(pro_seq), 12))
        for i in range(len(pro_seq)):
            pro_hot[i,] = self.one_of_k_encoding(pro_seq[i], self.pro_res_table)
            # pro_property[i,] = self.residue_features(pro_seq[i])
        return pro_hot
        # return np.concatenate((pro_hot, pro_property), axis=1)

    # target feature for target graph
    def PSSM_calculation(self, prot_id):
        aln_file = os.path.join(self.data_path, 'aln', prot_id + '.aln')
        pro_seq = self.id_to_sequence[prot_id]
        pfm_mat = np.zeros((len(self.pro_res_table), len(pro_seq)))

        with open(aln_file, 'r') as f:
            line_count = len(f.readlines())
            # f.seek(0)
            for line in f.readlines():
                if len(line) != len(pro_seq) + 1:
                    print('error', len(line), len(pro_seq))
                    continue
                count = 0
                for i in range(len(line) - 1):
                    res = line[i]
                    if res not in self.pro_res_table:
                        count += 1
                        continue
                    pfm_mat[self.pro_res_table.index(res), count] += 1
                    count += 1
        # print('pfm_mat = ', pfm_mat)
        # ppm_mat = pfm_mat / float(line_count)
        pseudocount = 0.8
        ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
        pssm_mat = ppm_mat
        # k = float(len(pro_res_table))
        # pwm_mat = np.log2(ppm_mat / (1.0 / k))
        # pssm_mat = pwm_mat

        return pssm_mat


def load_data():
    protein_feature_manager = ProteinFeatureManager('..\\..\\data\\Davis')
    raw_train_data = pd.read_csv('../../data/Davis/' + 'split_data_v1/prot_cold_start/train_fold0.csv')
    cp_squence = raw_train_data['PROTEIN_SEQUENCE'].values
    return protein_feature_manager, cp_squence


def calc(protein_feature_manager, c):
    for i, t in enumerate(protein_feature_manager.pro_res_table):
        if c == t:
            return i


def calc_1hop_nodes():
    seq_mp = collections.Counter(cp_squence)

    # mp = collections.defaultdict(set)
    mp = set()
    for s in seq_mp:
        pt_g = protein_feature_manager.get_contact_map(s)
        for u in networkx.nodes(pt_g):
            tmp_st = []
            for v in list(networkx.neighbors(pt_g, u)):
                tmp_st.append(s[v])

            tmp_st = frozenset(tmp_st)
            mp.add(tmp_st)
        print('finished -> ', s)
    return mp


if __name__ == "__main__":
    import collections

    protein_feature_manager, cp_squence = load_data()
    mp = calc_1hop_nodes()
    total = len(mp)
    # for s in mp:
    #     print('s = ', s, 'len = ', len(mp[s]))
    #     total += len(mp[s])
    print('total num = ', total)

    # for i, c in enumerate(s):
    #     lst = []
    #     for j, v in enumerate(ct_mp[i]):
    #         if v == 1:
    #             lst.append(j)
    #     lst = tuple(lst)
    #     mp[c].add(lst)

    # for key in mp:
    #     print('l = ', len(mp[key]))

    # node_features = protein_feature_manager.get_node_features(s)
    # avg_len += node_features.shape[0]
    # for k in mp:
    #     if k > 700:
    #         print(k)
    # print(avg_len // len(cp_squence))
    # test_sequence = ('MDSRAQLWGLALNKRRATLPHPGGSTNLKADPEELFTKLEKIGKGSFGEVFKGIDNRTQKVVAIKIIDLEEAEDEIEDIQQEITVLSQCDSPYVTKYYGSYLKDTKLWIIMEYLGGGSALDLLEPGPLDETQIATILREILKGLDYLHSEKKIHRDIKAANVLLSEHGEVKLADFGVAGQLTDTQIKRNTFVGTPFWMAPEVIKQSAYDSKADIWSLGITAIELARGEPPHSELHPMKVLFLIPKNNPPTLEGNYSKPLKEFVEACLNKEPSFRPTAKELLKHKFILRNAKKTSYLTELIDRYKRWKAEQSHDDSSSEDSDAETDGQASGGSDSGDWIFTIREKDPKNLENGALQPSDLDRNKMKDIPKRPFSQCLSTIISPLFAELKEKSQACGGNLGSIEELRGAIYLAEEACPGISDTMVAQLVQRLQRYSLSGGGTSSH')
    # # test_sequence = ('MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEELELDEQQKKRLEAFLTQKAKVGELKDDDFERISELGAGNGGVVTKVQHRPSGLIMARKLIHLEIKPAIRNQIIRELQVLHECNSPYIVGFYGAFYSDGEISICMEHMDGGSLDQVLKEAKRIPEEILGKVSIAVLRGLAYLREKHQIMHRDVKPSNILVNSRGEIKLCDFGVSGQLIDSMANSFVGTRSYMAPERLQGTHYSVQSDIWSMGLSLVELAVGRYPIPPPDAKELEAIFGRPVVDGEEGEPHSISPRPRPPGRPVSGHGMDSRPAMAIFELLDYIVNEPPPKLPNGVFTPDFQEFVNKCLIKNPAERADLKMLTNHTFIKRSEVEEVDFAGWLCKTLRLNQPGTPTRTAV')
    # node_features = protein_feature_manager.get_node_features(test_sequence)
    # contact_map = protein_feature_manager.get_contact_map(test_sequence)
    # pretrained_embedding = protein_feature_manager.get_pretrained_embedding(test_sequence)
    # np.set_printoptions(threshold=np.inf)

    # print(len(test_sequence))
    # print(node_features.shape)
    # print(node_features)
    # print(contact_map[10])
    # print(contact_map.shape)
    # print(pretrained_embedding.shape)
