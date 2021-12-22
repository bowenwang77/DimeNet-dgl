import dgl
import torch
import torch.nn as nn


class SubstrateConvLayer(nn.Module):
    def __init__(self, o_dim, e_dim, use_bias=True):
        super(SubstrateConvLayer, self).__init__()
        self.o_dim = o_dim
        self.e_dim = e_dim
        self.node_feat_name = 'embedded_feat'
        self.edge_feat_name = 'embedded_edge'
        self.use_bias = use_bias

        # construction of the network
        self.conv = nn.Linear(in_features=2 * self.o_dim + self.e_dim,
                              out_features=2 * self.o_dim,
                              bias=use_bias)
        self.bn1 = nn.BatchNorm1d(num_features=2 * self.o_dim)
        self.act_core = nn.Softplus()
        self.act_filter = nn.Sigmoid()
        self.bn2 = nn.BatchNorm1d(num_features=self.o_dim)
        self.act_final = nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_normal_(self.conv.weight)

    def edge_convolution(self, edges):
        cat_feat = torch.concat([edges.dst[self.node_feat_name],
                                 edges.src[self.node_feat_name],
                                 edges.data[self.edge_feat_name]], dim=1)
        # cat_feat.shape == (num_edges, 2 * n_f_len + e_f_len)
        return {'messages': cat_feat}

    def mul_reduction(self, nodes):
        conv_feat = nodes.mailbox['messages']
        conv_filter, conv_core = conv_feat.chunk(2, dim=-1)
        # conv_filter.shape == (num_nodes, num_edges, n_f_len)
        # conv_core.shape == (num_nodes, num_edges, n_f_len)
        conv_filter = self.act_filter(conv_filter)
        conv_core = self.act_core(conv_core)
        conv_core = conv_core * conv_filter  # element-wise mul
        conv_core = torch.sum(conv_core, dim=1)
        # conv_core.shape == (num_nodes, n_f_len)
        return {'messages': conv_core}

    def forward(self, inputs):
        graph, n_inputs, e_inputs = inputs

        with graph.local_scope():
            graph.ndata[self.node_feat_name] = n_inputs
            graph.edata[self.edge_feat_name] = e_inputs

            graph.apply_edges(func=self.edge_convolution)
            cat_feat = graph.edata['messages']
            cat_feat = self.conv(cat_feat)
            cat_feat = self.bn1(cat_feat)
            # cat_feat.shape == (num_edges, 2 * n_f_len)

            graph.edata.update({'messages': cat_feat})
            graph.update_all(message_func=dgl.function.copy_edge('messages', 'messages'),
                             reduce_func=self.mul_reduction)
            node_feat = graph.ndata.pop(self.node_feat_name)
            node_feat = self.bn2(node_feat)
            node_feat = self.act_final(node_feat + n_inputs)
            return node_feat


class SubstrateConvBlock(nn.Module):
    def __init__(self, i_dim, o_dim, e_dim, h_dim, num_layer=3, n_h=3,
                 use_bias=True):
        super(SubstrateConvBlock, self).__init__()

        # orig_atom_feat_dim -> atom_feat_dim
        self.linear_before = nn.Linear(i_dim, o_dim)
        self.conv_l = nn.ModuleList([SubstrateConvLayer(o_dim=o_dim,
                                                        e_dim=e_dim,
                                                        use_bias=use_bias)
                                     for _ in range(num_layer)])

        # atom_feat_dim -> hidden_dim
        self.linear_after = nn.Linear(o_dim, h_dim)
        self.act_after = nn.Softplus()
        if n_h - 1 > 0:
            self.linear_hidden = nn.ModuleList([nn.Linear(h_dim, h_dim)
                                                for _ in range(n_h - 1)])
            self.act_hidden = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])
        self.output_layer = nn.Linear(h_dim, 1)
        return

    def forward(self, inputs):
        graph, n_inputs, e_inputs = inputs
        n_inputs = self.linear_before(n_inputs)

        for conv in self.conv_l:
            n_inputs = conv((graph, n_inputs, e_inputs))

        graph.ndata['embedded_feat'] = n_inputs
        output = dgl.readout_nodes(graph, 'embedded_feat', op='mean')

        output = self.linear_after(output)
        output = self.act_after(output)
        if hasattr(self, 'linear_hidden'):
            for l_h, a_h in zip(self.linear_hidden, self.act_hidden):
                output = a_h(l_h(output))
        output = self.output_layer(output)
        return output
