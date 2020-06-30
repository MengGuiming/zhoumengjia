"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
SEED = 5
GCN_DROPOUT = 0.3
np.random.seed(SEED)
torch.manual_seed(SEED)


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, in_dim, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = in_dim

        self.in_drop = nn.Dropout(GCN_DROPOUT)
        self.gcn_drop = nn.Dropout(GCN_DROPOUT)

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, token_encode):
        '''
        :param adj:  batch, seqlen, seqlen
        :param token_encode: batch, seqlen, dm
        :return:
        '''
        # print('adj', adj.shape)
        # print('token_encode', token_encode.shape)
        # print('W[l]', self.W[0].weight.shape)
        embs = self.in_drop(token_encode)

        gcn_inputs = embs

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs, mask
