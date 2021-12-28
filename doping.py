"""QM9 dataset for graph property prediction (regression)."""
import os
import numpy as np
import scipy.sparse as sp
import torch
import dgl
import torch.nn.functional as F

from tqdm import trange
from dgl.data import QM9Dataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.convert import graph as dgl_graph

class DopingDataset(QM9Dataset):
    r"""
    """


    def __init__(self,
                 label_keys,
                 edge_funcs=None,
                 cutoff=5.0,
                 raw_dir=None,
                 force_reload=False,
                 verbose=False):

        self.edge_funcs = edge_funcs
        # self._keys = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        self._keys = ['Energy']
        super(DopingDataset, self).__init__(label_keys=label_keys,
                                  cutoff=cutoff,
                                  raw_dir=raw_dir,
                                  force_reload=force_reload,
                                  verbose=verbose)

    def has_cache(self):
        """ step 1, if True, goto step 5; else goto download(step 2), then step 3"""
        graph_path = f'{self.save_path}/dgl_Mat_graph.bin'
        line_graph_path = f'{self.save_path}/dgl_Mat_line_graph.bin'
        return os.path.exists(graph_path) and os.path.exists(line_graph_path)
        # return False

    def process(self):
        """ step 3 """
        # npz_path = f'{self.raw_dir}/qm9_eV.npz'
        npz_path = f'{self.raw_dir}/Material.npz'
        data_dict = np.load(npz_path, allow_pickle=True)
        # data_dict['N'] contains the number of atoms in each molecule,
        # data_dict['R'] consists of the atomic coordinates,
        # data_dict['Z'] consists of the atomic numbers.
        # Atomic properties (Z and R) of all molecules are concatenated as single tensors,
        # so you need this value to select the correct atoms for each molecule.
        self.N = data_dict['N']
        self.R = data_dict['R']
        self.Z = data_dict['Z']
        self.lattice=data_dict['lattice']
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        # graph labels
        self.label_dict = {}
        for k in self._keys:
            self.label_dict[k] = torch.tensor(data_dict[k], dtype=torch.float32)

        self.label = torch.stack([self.label_dict[key] for key in self.label_keys], dim=1)
        # graphs & features
        self.graphs, self.line_graphs = self._load_graph()
    
    def _load_graph(self):
        num_graphs = self.label.shape[0]
        graphs = []
        line_graphs = []
        
        for idx in trange(num_graphs):
            n_atoms = self.N[idx]
            # get all the atomic coordinates of the idx-th molecular graph
            R = self.R[self.N_cumsum[idx]:self.N_cumsum[idx + 1]]
            # calculate the distance between all atoms
            dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            # keep all edges that don't exceed the cutoff and delete self-loops
            adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(n_atoms, dtype=np.bool)
            adj = adj.tocoo()
            u, v = torch.tensor(adj.row), torch.tensor(adj.col)
            g = dgl_graph((u, v))
            g.ndata['R'] = torch.tensor(R, dtype=torch.float32)
            g.ndata['Z'] = torch.tensor(self.Z[self.N_cumsum[idx]:self.N_cumsum[idx + 1]], dtype=torch.long)
            
            # add user-defined features
            if self.edge_funcs is not None:
                for func in self.edge_funcs:
                    g.apply_edges(func)


            #####For Infinite plate 

            # calculate the distance between all atoms
            for i,j in [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]:
                R_inf=R+self.lattice[idx,:3]*i+self.lattice[idx,3:6]*j
                dist = np.linalg.norm(R[:, None, :] - R_inf[None, :, :], axis=-1)
                # keep all edges that don't exceed the cutoff 
                adj = sp.csr_matrix(dist <= self.cutoff)
                adj = adj.tocoo()
                u, v = torch.tensor(adj.row), torch.tensor(adj.col)

                R_src,R_dst=torch.from_numpy(R[u]).to(torch.float32),torch.from_numpy(R_inf[v]).to(torch.float32)
                dist_inf = torch.sqrt(F.relu(torch.sum((R_src - R_dst) ** 2, -1)))
                g.add_edges(u,v,{'d':dist_inf.reshape(u.shape[0]),'o':(R_src-R_dst).reshape(u.shape[0],3)})

            graphs.append(g)
            l_g = dgl.line_graph(g, backtracking=False)
            line_graphs.append(l_g)
    
        return graphs, line_graphs

    def save(self):
        """ step 4 """
        graph_path = f'{self.save_path}/dgl_Mat_graph.bin'
        line_graph_path = f'{self.save_path}/dgl_Mat_line_graph.bin'
        save_graphs(str(graph_path), self.graphs, self.label_dict)
        save_graphs(str(line_graph_path), self.line_graphs)

    def load(self):
        """ step 5 """
        graph_path = f'{self.save_path}/dgl_Mat_graph.bin'
        line_graph_path = f'{self.save_path}/dgl_Mat_line_graph.bin'
        self.graphs, label_dict = load_graphs(graph_path)
        self.line_graphs, _ = load_graphs(line_graph_path)
        self.label = torch.stack([label_dict[key] for key in self.label_keys], dim=1)

    def __getitem__(self, idx):
        r""" Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index
        
        Returns
        -------
        dgl.DGLGraph
            The graph contains:
            - ``ndata['R']``: the coordinates of each atom
            - ``ndata['Z']``: the atomic number
        Tensor
            Property values of molecular graphs
        """
        return self.graphs[idx], self.line_graphs[idx], self.label[idx]
