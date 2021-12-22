import dgl
import json
import torch
import numpy as np
from operator import itemgetter
from dgl.data.utils import load_graphs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from data_origi.data_preprocessing.ele_feat_embeddeding import ElementFeatDict


class GraphLoader:
    def __init__(self,
                 graph_type,
                 bool_expn,
                 one_hot=False,
                 graph_path='.',
                 prime_path=r'.\primitive',
                 load_ratio=1.0,
                 cgcnn_feat=False):

        assert graph_type in ['relaxation', 'direct', 'cluster']
        assert load_ratio <= 1.0

        expn = 'supercell' if bool_expn else 'prime'
        graph_p = graph_path + '\\' + f'graph_dataset_{graph_type}_{expn}.bin'
        self.graph_list, self.labels = load_graphs(graph_p)
        self.labels = self.labels['E']
        if load_ratio < 1.0:
            load_num = int(len(self.graph_list) * load_ratio)
            self.graph_list = self.graph_list[:load_num]
            self.labels = self.labels[:load_num]
        print('Finish loading graph data!')

        if cgcnn_feat:
            with open(graph_path + '\\' + 'atom_init.json') as f:
                efd = json.loads(f.read())
            self.node_feat_dict = {int(k): v for k, v in efd.items()}
        else:
            efd = ElementFeatDict(prime_path=prime_path)
            self.node_feat_dict = efd.generate_dict(one_hot=one_hot)
        print('Finish loading element features!')

        # container for graph datasets
        self.graph_train = []
        self.y_train = []
        self.feat_train = []
        self.edge_train = []

        self.graph_val = []
        self.y_val = []
        self.feat_val = []
        self.edge_val = []

        self.graph_test = []
        self.y_test = []
        self.feat_test = []
        self.edge_test = []

        return

    def tt_split(self, random_state=None, val_ratio=0.2, test_ratio=0.2,
                 normalize=False, standardize=False):

        """ Divide the dataset into training, validation and test parts;
            standardize if needed. """

        np.random.seed(random_state)
        samples = np.arange(0, len(self.graph_list))
        np.random.shuffle(samples)
        val_size = int(len(self.graph_list) * val_ratio)
        test_size = int(len(self.graph_list) * test_ratio)

        def choice_from_samples(sample_idx, size):
            sample_idx_idx = np.arange(0, len(sample_idx))
            sampled_idx_idx = np.random.choice(sample_idx_idx, size=size, replace=False)
            sampled_idx = sample_idx[sampled_idx_idx]
            left_idx = np.delete(sample_idx, sampled_idx_idx)
            return sampled_idx, left_idx

        def collect(sampled_idx, graph_list, y_list, feat_dict):
            sampled_graphs = list(itemgetter(*sampled_idx)(graph_list))
            sampled_y = y_list[sampled_idx]
            g = dgl.batch(sampled_graphs)
            sampled_feat = g.ndata['atom_num'].numpy()
            sampled_feat = list(map(feat_dict.get, sampled_feat))
            sampled_feat = np.array(sampled_feat)
            sampled_edge_feat = g.edata['edge_feat'][:, -1].numpy()
            sampled_edge_feat = self.gaussian_expn(sampled_edge_feat)
            return sampled_graphs, sampled_y, sampled_feat, sampled_edge_feat

        test_idx, samples = choice_from_samples(samples, test_size)
        val_idx, train_idx = choice_from_samples(samples, val_size)

        graph_object = self.graph_list
        self.graph_train, self.y_train, self.feat_train, self.edge_train = collect(
            train_idx, graph_object, self.labels, self.node_feat_dict)
        self.graph_val, self.y_val, self.feat_val, self.edge_val = collect(
            val_idx, graph_object, self.labels, self.node_feat_dict)
        self.graph_test, self.y_test, self.feat_test, self.edge_test = collect(
            test_idx, graph_object, self.labels, self.node_feat_dict)

        if normalize:
            ms = MinMaxScaler()
            self.feat_train = ms.fit_transform(self.feat_train)
            self.feat_val = ms.transform(self.feat_val)
            self.feat_test = ms.transform(self.feat_test)

        if standardize:
            ss = StandardScaler()
            self.feat_train = ss.fit_transform(self.feat_train)
            self.feat_val = ss.transform(self.feat_val)
            self.feat_test = ss.transform(self.feat_test)

        print('Finish splitting!')
        return self

    def batch(self, batch_sz=16, batch_sz_t=16):

        def index_feat(g_list):
            # sections for graph features
            node_num_list = []
            edge_num_list = []
            for g in g_list:
                node_num_list.append(g.num_nodes())
                edge_num_list.append(g.num_edges())
            node_feat_idx = np.cumsum(node_num_list)
            node_feat_idx = np.concatenate([np.zeros(shape=(1, )), node_feat_idx[:-1]])
            edge_feat_idx = np.cumsum(edge_num_list)
            edge_feat_idx = np.concatenate([np.zeros(shape=(1, )), edge_feat_idx[:-1]])
            return node_feat_idx.astype(int), edge_feat_idx.astype(int)

        def get_batch(g_list, f_list, e_list, y_list, batch_size):
            # sections for graphs and ys
            num_split = int(len(g_list) // batch_size) + 1
            g_b_list = []
            y_b_list = []
            f_b_list = []
            e_b_list = []
            f_b_idx, e_b_idx = index_feat(g_list)
            for i in range(num_split):
                start = i * batch_size
                end = (i + 1) * batch_size
                if end < len(g_list):
                    g_b_list.append(g_list[start: end])
                    y_b_list.append(y_list[start: end])
                    f_b_list.append(f_list[f_b_idx[start]: f_b_idx[end]])
                    e_b_list.append(e_list[e_b_idx[start]: e_b_idx[end]])
                else:
                    if start >= len(g_list):
                        continue
                    g_b_list.append(g_list[start:])
                    y_b_list.append(y_list[start:])
                    f_b_list.append(f_list[f_b_idx[start]:])
                    e_b_list.append(e_list[e_b_idx[start]:])
            return g_b_list, f_b_list, e_b_list, y_b_list

        def dataset(g_b_list, f_b_list, e_b_list, y_b_list):
            dataset_l = []
            for i in range(len(g_b_list)):
                if len(g_b_list[i]) == 0:
                    continue
                try:
                    g_b = dgl.batch(g_b_list[i]).to('/gpu:0')
                except RuntimeError:
                    g_b = dgl.batch(g_b_list[i])
                f_b = torch.FloatTensor(f_b_list[i])
                e_b = torch.FloatTensor(e_b_list[i])
                y_b = torch.FloatTensor(y_b_list[i])
                dataset_l.append(((g_b, f_b, e_b), y_b))
            return dataset_l

        graph_train_b, data_train_b, e_train_b, y_train_b = get_batch(
            self.graph_train, self.feat_train, self.edge_train, self.y_train, batch_sz)
        graph_val_b, data_val_b, e_val_b, y_val_b = get_batch(
            self.graph_val, self.feat_val, self.edge_val, self.y_val, batch_sz_t)
        graph_test_b, data_test_b, e_test_b, y_test_b = get_batch(
            self.graph_test, self.feat_test, self.edge_test, self.y_test, batch_sz_t)

        dataset_train = dataset(graph_train_b, data_train_b, e_train_b, y_train_b)
        dataset_val = dataset(graph_val_b, data_val_b, e_val_b, y_val_b)
        dataset_test = dataset(graph_test_b, data_test_b, e_test_b, y_test_b)

        print('Finish batching!')
        return dataset_train, dataset_val, dataset_test

    def gaussian_expn(self, dist, n_basis=40, rcut=4.0, gamma=1.0):
        dist = np.expand_dims(dist, axis=1)
        # dist.shape == (*, 1)
        dist_mat = np.tile(dist, (1, n_basis))
        # dist_mat.shape == (*, num_basis)

        miu = np.linspace(0.0, rcut, n_basis)
        miu_mat = np.expand_dims(miu, axis=0)
        # miu.shape == (1, num_basis)
        miu_mat = np.tile(miu_mat, (dist.shape[0], 1))
        # miu.shape == (*, num_basis)

        expn = dist_mat - miu_mat
        dist_shift = - gamma * (expn ** 2)
        g_emb = np.exp(dist_shift)
        # g_emb.shape == (*, num_basis)

        return g_emb


if __name__ == '__main__':
    import time
    s = time.process_time()
    gl = GraphLoader(bool_expn=False, graph_type='direct', one_hot=True)
    tr_g, v_g, te_g = gl.tt_split().batch(batch_sz=32, batch_sz_t=32)
    print(f'Time used: {time.process_time() - s}')
