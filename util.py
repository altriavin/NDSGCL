from collections import defaultdict
from re import split
from random import shuffle, choice
import torch.nn.functional as F

import numpy as np
import scipy as sp
import torch


class ModelConf(object):
    def __init__(self, file):
        self.config = {}
        self.read_configuration(file)

    def __getdrug__(self, drug):
        return self.config[drug]

    def contain(self, key):
        return key in self.config

    def read_configuration(self, file):
        with open(file) as f:
            for ind, line in enumerate(f):
                if line.strip() != '':
                    key, value = line.strip().split('=')
                    self.config[key] = value


class OptionConf(object):
    def __init__(self, content):
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i, drug in enumerate(self.line):
            if (drug.startswith('-') or drug.startswith('--')) and not drug[1:].isdigit():
                ind = i+1
                for j, sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:])-1:
                        ind = j + 1
                        break
                try:
                    self.options[drug] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[drug] = 1

    def __getitem__(self, drug):
        return self.options[drug]

    def keys(self):
        return self.options.keys()

    def is_main_on(self):
        return self.mainOption

    def contain(self, key):
        return key in self.options


def next_batch_pairwise(data, batch_size):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            lncRNAs = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            drugs = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            lncRNAs = [training_data[idx][0] for idx in range(batch_id, data_size)]
            drugs = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        drug_list = list(data.drug.keys())
        for i, lncRNA in enumerate(lncRNAs):
            i_idx.append(data.drug[drugs[i]])
            u_idx.append(data.lncRNA[lncRNA])
            neg_drug = choice(drug_list)
            while neg_drug in data.training_set_u[lncRNA]:
                neg_drug = choice(drug_list)
            j_idx.append(data.drug[neg_drug])
        yield u_idx, i_idx, j_idx


def bpr_loss(lncRNA_emb, pos_drug_emb, neg_drug_emb):
    pos_score = torch.mul(lncRNA_emb, pos_drug_emb).sum(dim=1)
    neg_score = torch.mul(lncRNA_emb, neg_drug_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg


def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


def load_data_set(file):
    data = []
    with open(file) as f:
        for line in f:
            drugs = split(' ', line.strip())
            lncRNA_id = drugs[0]
            drug_id = drugs[1]
            weight = 1.0
            data.append([int(lncRNA_id), int(drug_id), float(weight)])
    return data


def convert_sparse_mat_to_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)


class Data(object):
    def __init__(self, conf, training, test):
        self.config = conf
        self.training_data = training[:]
        self.test_data = test[:]


class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat


class Interaction(Data, Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.lncRNA = {}
        self.drug = {}
        self.id2lncRNA = {}
        self.id2drug = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_drug = set()
        self.__generate_set()
        self.lncRNA_num = len(self.training_set_u)
        self.drug_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __generate_set(self):
        for entry in self.training_data:
            lncRNA, drug, rating = entry
            if lncRNA not in self.lncRNA:
                self.lncRNA[lncRNA] = len(self.lncRNA)
                self.id2lncRNA[self.lncRNA[lncRNA]] = lncRNA
            if drug not in self.drug:
                self.drug[drug] = len(self.drug)
                self.id2drug[self.drug[drug]] = drug
                # lncRNAList.append
            self.training_set_u[lncRNA][drug] = rating
            self.training_set_i[drug][lncRNA] = rating
        for entry in self.test_data:
            lncRNA, drug, rating = entry
            if lncRNA not in self.lncRNA:
                continue
            self.test_set[lncRNA][drug] = rating
            self.test_set_drug.add(drug)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.lncRNA_num + self.drug_num
        row_idx = [self.lncRNA[pair[0]] for pair in self.training_data]
        col_idx = [self.drug[pair[1]] for pair in self.training_data]
        lncRNA_np = np.array(row_idx)
        drug_np = np.array(col_idx)
        ratings = np.ones_like(lncRNA_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (lncRNA_np, drug_np + self.lncRNA_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (lncRNA_np_keep, drug_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (lncRNA_np_keep, drug_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.lncRNA[pair[0]]]
            col += [self.drug[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.lncRNA_num,self.drug_num),dtype=np.float32)
        return interaction_mat

    def get_lncRNA_id(self, u):
        if u in self.lncRNA:
            return self.lncRNA[u]

    def get_drug_id(self, i):
        if i in self.drug:
            return self.drug[i]

    def training_size(self):
        return len(self.lncRNA), len(self.drug), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_drug), len(self.test_data)

    def contain(self, u, i):
        if u in self.lncRNA and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_lncRNA(self, u):
        if u in self.lncRNA:
            return True
        else:
            return False

    def contain_drug(self, i):
        if i in self.drug:
            return True
        else:
            return False

    def lncRNA_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def drug_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2lncRNA[u]
        k, v = self.lncRNA_rated(u)
        vec = np.zeros(len(self.drug))
        for pair in zip(k, v):
            iid = self.drug[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2drug[i]
        k, v = self.drug_rated(i)
        vec = np.zeros(len(self.lncRNA))
        for pair in zip(k, v):
            uid = self.lncRNA[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.lncRNA), len(self.drug)))
        for u in self.lncRNA:
            k, v = self.lncRNA_rated(u)
            vec = np.zeros(len(self.drug))
            for pair in zip(k, v):
                iid = self.drug[pair[0]]
                vec[iid] = pair[1]
            m[self.lncRNA[u]] = vec
        return m