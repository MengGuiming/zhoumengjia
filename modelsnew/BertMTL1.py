import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
from modelsnew.attention import SimpleEncoder
from pytorch_transformers import *
from torch.nn.utils.rnn import pad_sequence
from modelsnew.GCN import GCN
import time

path = "./modeling_bert"


class BertMTL1(nn.Module):
    def __init__(self, config):
        super(BertMTL1, self).__init__()
        self.config = config

        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))

        self.word_emb.weight.requires_grad = False
        self.use_entity_type = True
        self.use_coreference = True
        self.use_distance = True

        self.hidden_size = 120
        bert_hidden_size = 768
        input_size = config.data_word_vec.shape[1]
        if self.use_entity_type:
            input_size += config.entity_type_size
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        if self.use_coreference:
            input_size += config.coref_size
            # self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
            self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

        # input_size += char_hidden

        self.input_size = input_size
        modelConfig = BertConfig.from_pretrained(path + "/" + "bert-base-uncased-config.json")
        self.bert = BertModel.from_pretrained(
            path + "/" + 'bert-base-uncased-pytorch_model.bin', config=modelConfig)
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.linear_re = nn.Linear(bert_hidden_size+config.coref_size+config.entity_type_size, hidden_size)


        # self.ent_att_enc = SimpleEncoder(hidden_size*2, 4, 1)
        self.linear1 = nn.Linear(bert_hidden_size, self.hidden_size, bias=False)
        self.linear2 = nn.Linear(bert_hidden_size, self.hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.induction = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size)).cuda()
        self.linear3 = nn.Linear(bert_hidden_size, 1, bias=False)

        self.gcn_layer = GCN(in_dim=bert_hidden_size, mem_dim=self.hidden_size, num_layers=2)
        # self.biaffine = Biaffine(hidden_size, hidden_size, config.relation_num, (True, True))
        self.classify_weight = nn.Parameter(torch.FloatTensor(self.hidden_size, config.relation_num, self.hidden_size)).cuda()
        self.classify_bias = nn.Parameter(torch.FloatTensor(config.relation_num)).cuda()


    def mask_lengths(self, batch_size, doc_size, lengths):
        masks = torch.ones(batch_size, doc_size).cuda()
        index_matrix = torch.arange(0, doc_size).expand(batch_size, -1).cuda()
        index_matrix = index_matrix.long()
        # doc_lengths = torch.tensor(lengths).view(-1,1)
        doc_lengths = lengths.view(-1, 1)
        doc_lengths_matrix = doc_lengths.expand(-1, doc_size)
        masks[torch.ge(index_matrix - doc_lengths_matrix, 0)] = 0
        return masks

    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, node_mapping, node_mask,
                relation_mask, dis_h_2_t, dis_t_2_h, sent_idxs, sent_lengths, reverse_sent_idxs, context_masks,
                context_starts, sent_mask,
                sent_mapping):

        context_output = self.bert(context_idxs, attention_mask=context_masks)[0]
        context_output = [layer[starts.nonzero().squeeze(1)]
                          for layer, starts in zip(context_output, context_starts)]
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)
        context_output = torch.nn.functional.pad(context_output,
                                                 (0, 0, 0, context_idxs.size(-1) - context_output.size(-2)))

        node_rep = torch.matmul(node_mapping, context_output)
        output_linear1 = self.tanh(self.linear1(node_rep))
        output_linear2 = self.tanh(self.linear2(node_rep))
        output_bilinear = torch.matmul(output_linear1, self.induction)
        output_bilinear = torch.matmul(output_bilinear, torch.transpose(output_linear2,1,2))
        output_linear3 = self.linear3(node_rep)
        node_num = output_bilinear.shape[1]
        mask1 = torch.eye(node_num).unsqueeze(0).repeat(output_bilinear.shape[0], 1, 1).cuda()
        P_edge = output_bilinear = torch.where(mask1 == 1, torch.tensor(0.0).cuda(), torch.exp(output_bilinear))
        mask2 = torch.ones(output_bilinear.shape[0], 1, node_num).cuda()
        temp = torch.matmul(mask2, output_bilinear).repeat(1, node_num, 1)
        output_bilinear = torch.where(mask1 == 1, temp, -output_bilinear)
        output_bilinear[:, 1, :] = output_linear3.squeeze(2)
        output_bilinear = [t.inverse() for t in torch.unbind(output_bilinear)]
        output_bilinear = torch.stack(output_bilinear).cuda()
        temp1 = torch.matmul(P_edge, output_bilinear)
        temp2 = torch.matmul(P_edge, torch.transpose(output_bilinear, 1, 2))
        edge_output = temp1 - temp2
        edge_output[:, 1, 1] = torch.tensor(0.0).cuda()
        edge_output[:, 1, :] = temp1[:, 1, :]
        edge_output[:, :, 1] = -temp2[:, :, 1]
        gcn_output, gcn_mask = self.gcn_layer(edge_output, node_rep)
        entity_num = relation_mask.shape[1]
        entity_output = torch.zeros(relation_mask.shape[0], entity_num, gcn_output.shape[-1]).cuda()
        for it, t in enumerate(torch.unbind(gcn_output)):
            select = torch.eq(node_mask[it], 1).nonzero().unbind(dim=1)
            entity_output[it, :t[select].shape[0]] = t[select]
        # predict_re = self.biaffine(entity_output, entity_output)
        batch = relation_mask.shape[0]
        predict_re = torch.matmul(entity_output, self.classify_weight.view(self.hidden_size, -1)).view(batch, -1, self.hidden_size)
        predict_re = torch.matmul(predict_re, torch.transpose(entity_output, 1, 2)).view(batch, entity_num, self.config.relation_num, entity_num)
        # predict_re = torch.transpose(predict_re,1,2) + self.classify_bias
        predict_re = predict_re.permute(0, 1, 3, 2) + self.classify_bias
        return predict_re


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'



class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, hidden = self.rnns[i](output, hidden)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        lengths = torch.tensor(input_lengths)
        lens, indices = torch.sort(lengths, 0, True)
        input = input[indices]
        _, _indices = torch.sort(indices, 0)

        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        # if input_lengths is not None:
        #    lens = input_lengths.data.cpu().numpy()
        lens[lens == 0] = 1

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, (hidden, c) = self.rnns[i](output, (hidden, c))

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        for i, output in enumerate(outputs):
            outputs[i] = output[_indices]
        # if self.concat:
        #     return torch.cat(outputs, dim=-1)
        return outputs


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input * output_one, output_two * output_one], dim=-1)
