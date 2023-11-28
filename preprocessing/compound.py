# -*- encoding: utf-8 -*-
'''
@Time       : 2022/11/26 17:19
@Author     : Tian Tan
@Email      :
@File       : compound.py
@Project    :
@Description:
'''
import numpy as np
import torch_geometric
import torch_sparse
# from mol2vec.features import (DfVec, MolSentence, mol2alt_sentence,
#                               mol2sentence)
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys, AllChem
from torch_geometric.data import Data

from .const import (ALLEN_NEGATIVITY_TABLE, ATOM_CLASS_TABLE,
                    ELECTRON_AFFINITY_TABLE, ELEMENT_LIST, NUM_ATOM_FEAT, PT)
from .utils import one_of_k_encoding, one_of_k_encoding_unk
import torch
import torch_geometric.utils as utils
from gensim.models import Word2Vec, word2vec
from . import algos
import os
import pandas as pd

class CompoundData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''
    def __inc__(self, key, value, *args, **kwargs):
    # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
    # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        if key == 'subgraph_edge_index':
            return self.num_subgraph_nodes
        if key == 'subgraph_node_index':
            return self.num_nodes
        if key == 'subgraph_indicator':
            return self.num_nodes
        if 'index' in key:
            return self.num_nodes
        else:
            return 0
        # return super().__inc__(key, value, *args, **kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"


def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
    ]  # 6-dim

    results = (one_of_k_encoding_unk(atom.GetSymbol(), symbol) + one_of_k_encoding(atom.GetDegree(), degree) +
               [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
               one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]
               )  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'),
                                                      ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except Exception:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results


def atomic_features(atomic_num):
    # Symbol
    # symbol = PT.GetElementSymbol(atomic_num)
    # symbol_k = one_of_k_encoding_unk(symbol, ELEMENT_LIST)

    # Period
    outer_electrons = PT.GetNOuterElecs(atomic_num)
    outer_electrons_k = one_of_k_encoding(outer_electrons, list(range(0, 8 + 1)))

    # Default Valence
    default_electrons = PT.GetDefaultValence(atomic_num)  # -1 for transition metals
    default_electrons_k = one_of_k_encoding(default_electrons, list(range(-1, 8 + 1)))

    # Orbitals / Group / ~Row
    orbitals = next(j + 1 for j, val in enumerate([2, 10, 18, 36, 54, 86, 120]) if val >= atomic_num)
    orbitals_k = one_of_k_encoding(orbitals, list(range(0, 7 + 1)))

    # IUPAC Series
    atom_series = ATOM_CLASS_TABLE[atomic_num]
    atom_series_k = one_of_k_encoding(atom_series, list(range(0, 9 + 1)))

    # Centered Electrons
    centered_oec = abs(outer_electrons - 4)

    # Electronegativity & Electron Affinity
    try:
        allen_electronegativity = ALLEN_NEGATIVITY_TABLE[atomic_num]
    except KeyError:
        allen_electronegativity = 0
    try:
        electron_affinity = ELECTRON_AFFINITY_TABLE[atomic_num]
    except KeyError:
        electron_affinity = 0

    # Mass & Radius (van der waals / covalent / bohr 0)
    floats = [
        centered_oec, allen_electronegativity, electron_affinity,
        PT.GetAtomicWeight(atomic_num),
        PT.GetRb0(atomic_num),
        PT.GetRvdw(atomic_num),
        PT.GetRcovalent(atomic_num), outer_electrons, default_electrons, orbitals
    ]
    # print(symbol_k + outer_electrons_k + default_electrons_k + orbitals_k + atom_series_k + floats)

    # Compose feature array
    # feature_array = np.array(symbol_k + outer_electrons_k + default_electrons_k + orbitals_k + atom_series_k + floats,
    #                          dtype=np.float32)
    feature_array = np.array(outer_electrons_k + default_electrons_k + orbitals_k + atom_series_k + floats,
                             dtype=np.float32)
    # Cache in dict for future use
    return feature_array


def edge_features(bond):
    '''
    Get bond features
    '''
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    # adjacency_1hop = np.array(adjacency) + np.eye(adjacency.shape[0])
    return adjacency


def ori_get_mol_features(smiles, atom_dim):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        raise RuntimeError("SMILES cannot been parsed!")
    # mol = Chem.AddHs(mol)
    # (bond_i, bond_j, bond_features)
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol.GetBonds()])
    # Separate (bond_i, bond_j, bond_features) to (bond_i, bond_j) and bond_features
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (torch.LongTensor([]), torch.FloatTensor([]))
    # Convert the graph to undirect graph, e.g., [(1, 0)] to [(1, 0), (0, 1)]
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats]*2, dim=0) if len(edge_feats) else edge_feats

    atom_feat = torch.zeros((mol.GetNumAtoms(), atom_dim))
    atom_num = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        # atomic_features(atom.GetAtomicNum())
        # atom_feat[atom.GetIdx(), :] = np.append(atom_features(atom), atomic_features(atom.GetAtomicNum()))
        atom_feat[atom.GetIdx(), :] = torch.FloatTensor(atom_features(atom))
    # This is the most essential step to convert a node graph to a line graph
    # line_graph_edge_index = torch.LongTensor([])
    # if edge_list.nelement() != 0:
    #     conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
    #     line_graph_edge_index = conn.nonzero(as_tuple=False).T
    new_edge_index = edge_list.T
    return atom_feat, new_edge_index

def get_mol_features(smiles, atom_dim=34, k_hop=3, use_subgraph_edge_attr=False, k_hop_subgraph=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        raise RuntimeError("SMILES cannot been parsed!")
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    atom_feat = torch.zeros((mol.GetNumAtoms(), atom_dim))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = torch.FloatTensor(atom_features(atom))
        
    # import pdb; pdb.set_trace()
    if edge_list.shape[0] > 0:
        edge_index = edge_list.T
        degree = torch_geometric.utils.degree(edge_index[0])
        # degree = torch_geometric.utils.degree(edge_list.T[0])
        degree = 1 / torch.sqrt(degree)
    else:
        edge_index = edge_list
        degree = torch.zeros(mol.GetNumAtoms())
    


    if k_hop_subgraph:
        # for the k hop subgraph gnn
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
        subgraph_node_index = None

        # Each graph will become a block diagonal adjacency matrix of
        # all the k-hop subgraphs centered around each node. The edge
        # indices get augumented within a given graph to make this
        # happen (and later are augmented for proper batching)
        subgraph_edge_index = None

        # This identifies which indices correspond to which subgraph
        # (i.e. which node in a graph)
        subgraph_indicator_index = None

        # subgraph_edge_attr = None
        #
        # This gets the edge attributes for the new indices
        if use_subgraph_edge_attr:
            subgraph_edge_attr = None


        node_indices = []
        edge_indices = []
        edge_attributes = []
        indicators = []
        edge_index_start = 0

        for node_idx in range(atom_feat.shape[0]):
            sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                node_idx,
                k_hop,
                edge_index,
                relabel_nodes=True,
                num_nodes=atom_feat.shape[0]
            )
            node_indices.append(sub_nodes)
            edge_indices.append(sub_edge_index + edge_index_start)
            indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
            if use_subgraph_edge_attr and edge_feats is not None:
                edge_attributes.append(edge_feats[edge_mask])  # CHECK THIS DIDN"T BREAK ANYTHING
            edge_index_start += len(sub_nodes)

        subgraph_node_index = torch.cat(node_indices)
        subgraph_edge_index = torch.cat(edge_indices, dim=1)
        subgraph_indicator_index = torch.cat(indicators)
        subgraph_edge_attr = None
        if use_subgraph_edge_attr and edge_feats is not None:
            subgraph_edge_attr = (torch.cat(edge_attributes))

        num_subgraph_nodes = len(subgraph_node_index)

        N = atom_feat.size(0)
        edge_index_2hop, _ = torch_sparse.spspmm(edge_index, torch.ones([edge_index[1].size(0)], device='cpu'),
                                                 edge_index, torch.ones([edge_index[1].size(0)], device='cpu'),
                                                 N, N, N)
        edge_index_3hop, _ = torch_sparse.spspmm(edge_index_2hop,
                                                 torch.ones([edge_index_2hop[1].size(0)], device='cpu'),
                                                 edge_index, torch.ones([edge_index[1].size(0)], device='cpu'),
                                                 N, N, N)

        compound_graph = CompoundData(x=atom_feat, edge_index=edge_index, subgraph_node_index=subgraph_node_index
                                      , subgraph_edge_index=subgraph_edge_index
                                      , subgraph_indicator=subgraph_indicator_index.type(torch.LongTensor)
                                      , num_subgraph_nodes=num_subgraph_nodes, subgraph_edge_attr=subgraph_edge_attr
                                      , edge_index_2hop=edge_index_2hop, edge_index_3hop=edge_index_3hop)

    else:
        N = atom_feat.size(0)
        adj = torch.zeros([N, N], dtype=torch.bool)
        if edge_index.shape[0] > 0:
            adj[edge_index[0, :], edge_index[1, :]] = True
            shortest_path_result, path = algos.floyd_warshall(adj.numpy())
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            edge_index_2hop, _ = torch_sparse.spspmm(edge_index, torch.ones([edge_index[1].size(0)], device='cpu'),
                                              edge_index, torch.ones([edge_index[1].size(0)], device='cpu'),
                                              N, N, N)
            edge_index_3hop, _ = torch_sparse.spspmm(edge_index_2hop, torch.ones([edge_index_2hop[1].size(0)], device='cpu'),
                                                edge_index, torch.ones([edge_index[1].size(0)], device='cpu'),
                                                N, N, N)
        else:
            spatial_pos = torch.ones((N,N)) * 510
            edge_index = torch.zeros((2,1)).long()
            edge_index_2hop = edge_index
            edge_index_3hop = edge_index
        # spatial_pos = None

        
        # compound_word_emb = get_mol2vec_features(model, smiles)
        compound_graph = CompoundData(x=atom_feat, edge_index=edge_index, edge_index_2hop=edge_index_2hop
                                      , edge_index_3hop=edge_index_3hop)

        # produce fingerprint
        macc_fp = MACCSkeys.GenMACCSKeys(mol)  # 167
        rdk_fp = Chem.RDKFingerprint(mol, fpSize=1024)  # 1024
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)  # 1024
        avalon_fp = pyAvalonTools.GetAvalonFP(mol)  # 512


        mol_fp = torch.LongTensor(np.concatenate([macc_fp, morgan_fp, rdk_fp, avalon_fp]))


    return compound_graph, degree, mol_fp, spatial_pos


def sentences2vec(sentences, model, unseen=None):
    """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.

    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
    """

    keys = set(model.wv.key_to_index)
    vec = []

    if unseen:
        unseen_vec = model.wv.get_vector(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.get_vector(y) if y in set(sentence) & keys
                            else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & keys]))
    return np.array(vec)


# def get_mol2vec_features(model, smiles):
#     mol = Chem.MolFromSmiles(smiles)
#
#     sen = (MolSentence(mol2alt_sentence(mol, 0)))
#
#     mol_vec = (sentences2vec(sen, model, unseen='UNK'))
#
#     return mol_vec
class CompoundFeatureManager:
    def __init__(self, datapath):
        compound_smiles = pd.read_csv(os.path.join(datapath, 'compound_smiles.csv'))
        self.compound_smiles = compound_smiles['smiles']
        self.compound_fp = dict()
        self.compound_graph = dict()
        self.compound_degree = dict()
        self.compound_word_emb = dict()
        self.compound_spatial_pos = dict()
        self.get_mol_graph()
    def get_mol_graph(self):
        print("Extracting compound graph")
        for s in self.compound_smiles:
            s = Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
            self.compound_graph[s], self.compound_degree[s], compound_fp, compound_spatial_pos = get_mol_features(s)
            self.compound_spatial_pos[s] = compound_spatial_pos
            self.compound_fp[s] = compound_fp
        print('done')


if __name__ == "__main__":
    import pandas as pd

    # model = word2vec.Word2Vec.load("..\\..\\mol2vec\\model_300dim.pkl")
    raw_train_data = pd.read_csv('../../data/Davis/' + 'split_data_v1/prot_cold_start/train_fold0.csv')
    # cp =  raw_train_data['COMPOUND_SMILES'].values

    st = set()
    #
    for v in raw_train_data['COMPOUND_SMILES'].values:
        st.add(v)
    #
    compound_smiles = pd.DataFrame(list(st), columns=['smiles'])
    compound_smiles.to_csv('compound_smiles.csv', index=False)






    # for cp in compound_smiles['']:
    #     print(cp)



    # from rdkit.Chem import AllChem
    # from rdkit.Chem import DataStructs
    # import numpy as np
    # from rdkit import Chem
    #
    #
    #
    # # array = np.zeros((0,), dtype=np.int8)
    # # DataStructs.ConvertToNumpyArray(fp, array)
    #
    # avg_len = 0
    # mp = dict()
    # cnt_pos = list()
    # for s in cp:
    #     mp[s] = 1
    # from collections import Counter
    # avg = 0
    # for s in mp:
    #     mol = Chem.MolFromSmiles(s)
    #     features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    #     features = np.zeros((1,))
    #     # print(features)
    #     DataStructs.ConvertToNumpyArray(features_vec, features)
    #     feature_index = np.where(features == 1)
    #     avg += len(feature_index[0])
    #     cnt_pos.extend(feature_index[0])
    #
    # # print('avg = ', avg // len(mp))
    # cnt_pos = Counter(cnt_pos)
    # res = sorted(cnt_pos.items(), key=lambda x: -x[1])

    # for i, t in enumerate(res):
    #     if t[1] < 5:
    #         print(i)
    #         break

