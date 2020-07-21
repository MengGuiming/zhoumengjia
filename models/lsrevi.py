import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from models.encoder import Encoder, EncoderEntity
from models.attention import SelfAttention
from models.reasoner import DynamicReasoner
from models.reasoner import StructInduction

class LSREVI(nn.Module):
    def __init__(self, config):
        super(LSREVI, self).__init__()
        self.config = config

        self.finetune_emb = config.finetune_emb

        self.word_emb = nn.Embedding(config.data_word_vec.shape[0], config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        if not self.finetune_emb:
            self.word_emb.weight.requires_grad = False

        self.ner_emb = nn.Embedding(13, config.entity_type_size, padding_idx=0)

        self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

        hidden_size = config.rnn_hidden
        input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size #+ char_hidden

        self.linear_re = nn.Linear(hidden_size * 2,  hidden_size)

        self.linear_sent = nn.Linear(hidden_size * 2,  hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size, hidden_size, hidden_size)

        self.self_att = SelfAttention(hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size+config.dis_size,  hidden_size+config.dis_size, hidden_size)
        self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)

        self.linear_output = nn.Linear(2 * hidden_size, config.relation_num)

        self.relu = nn.ReLU()

        self.dropout_rate = nn.Dropout(config.dropout_rate)

        self.rnn_sent = Encoder(input_size, hidden_size, config.dropout_emb, config.dropout_rate)
        self.rnn_entity = EncoderEntity(config.data_word_vec.shape[1], hidden_size, 1, True, True, 1 - config.keep_prob, False)
        self.rnn_evidence = Encoder(config.data_word_vec.shape[1], hidden_size, config.dropout_emb, config.dropout_rate)

        self.score_linear1 = nn.Linear(2*hidden_size, 1)
        self.score_linear2 = nn.Linear(2*hidden_size, 1)

        self.hidden_size = hidden_size

        self.use_struct_att = config.use_struct_att
        if  self.use_struct_att == True:
            self.structInduction = StructInduction(hidden_size // 2, hidden_size, True)

        self.dropout_gcn = nn.Dropout(config.dropout_gcn)
        self.reasoner_layer_first = config.reasoner_layer_first
        self.reasoner_layer_second = config.reasoner_layer_second
        self.use_reasoning_block = config.use_reasoning_block
        if self.use_reasoning_block:
            self.reasoner = nn.ModuleList()
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_first, self.dropout_gcn))
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_second, self.dropout_gcn))

    def doc_encoder(self, input_sent, context_seg):
        """
        :param sent: sent emb
        :param context_seg: segmentation mask for sentences in a document
        :return:
        """
        batch_size = context_seg.shape[0]
        docs_emb = [] # sentence embedding
        docs_len = []
        sents_emb = []

        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = ((context_seg[batch_no] == 1).nonzero()).squeeze(-1).tolist() # array of start point for sentences in a document
            pre_index = 0
            for i, index in enumerate(sent_index):
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index:index+1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index+1:index+1])
                        sent_lens.append(index - pre_index)
                pre_index = index

            sents = pad_sequence(sent_list).permute(1,0,2)
            sent_lens_t = torch.LongTensor(sent_lens).cuda()
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_sent(sents, sent_lens_t) # sentence embeddings for a document.

            doc_emb = None
            for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
                if i == 0:
                    doc_emb = emb[:sen_len]
                else:
                    doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim = 0)

            docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))

        docs_emb = pad_sequence(docs_emb).permute(1,0,2) # B * # sentence * Dimention
        sents_emb = pad_sequence(sents_emb).permute(1,0,2)

        return docs_emb, sents_emb


    def select_sentences(self, input_sent, context_seg, input_lengths, h_mapping, t_mapping):
        """
        :param sent: sent emb
        :param context_seg: segmentation mask for sentences in a document
        :param h_mapping, t_mapping: entityA and entityB
        :return:
        """
        batch_size = context_seg.shape[0]
        h_t_num = h_mapping.shape[1]
        max_sent_len = 0
        # docs_emb = []  # sentence embedding
        docs_len = []
        sents_emb = []
        sent_indexs = []

        # entityA entityB
        h_t_len = h_mapping.shape[1]
        input_sent_ = input_sent.unsqueeze(1).repeat(1, h_t_len, 1, 1)
        doc_entities_emb = []

        # score
        all_scores = torch.zeros(batch_size, h_t_num, 50).cuda()
        all_scores_mask = torch.zeros(batch_size, h_t_num, 50).cuda()

        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = ((context_seg[batch_no] == 1).nonzero()).squeeze(
                -1).tolist()  # array of start point for sentences in a document
            sent_indexs.append(sent_index)
            pre_index = 0
            for i, index in enumerate(sent_index):
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index:index + 1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index + 1:index + 1])
                        sent_lens.append(index - pre_index)
                pre_index = index

            sents = pad_sequence(sent_list).permute(1, 0, 2)
            sent_lens_t = torch.LongTensor(sent_lens).cuda()
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_evidence(sents, sent_lens_t)  # sentence embeddings for a document.
            max_sent_len = max(sents.shape[0], max_sent_len)
            # doc_emb = None
            # for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
            #     if i == 0:
            #         doc_emb = emb[:sen_len]
            #     else:
            #         doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim=0)
            #
            # docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))
            entities_emb = []
            entity_lens = []
            for h_t_no in range(h_t_len):
                entityA_ = torch.ne(h_mapping[batch_no][h_t_no], 0.0)
                entityB_ = torch.ne(t_mapping[batch_no][h_t_no], 0.0)
                if entityA_.nonzero().shape[0] == 0 and entityB_.nonzero().shape[0] == 0:
                    break
                entityA = input_sent_[batch_no][h_t_no][entityA_.squeeze(-1)]
                entityB = input_sent_[batch_no][h_t_no][entityB_.squeeze(-1)]
                entities_emb += [torch.cat([entityA, entityB], dim=0)]
                entity_lens.append(entities_emb[h_t_no].shape[0])
            entity_lens = torch.tensor(entity_lens)
            entities_emb = pad_sequence(entities_emb).permute(1, 0, 2)
            entity_context = self.rnn_entity(entities_emb, entity_lens)
            doc_entities_emb.append(entity_context[-1])

            # score
            sents_con = sent_emb.squeeze(1)
            entity_con = entity_context[-1]
            for j,h_t_pair in enumerate(entity_con.unbind(dim=0)):
                score = torch.matmul(sents_con, h_t_pair.permute(1,0))
                score = torch.softmax(score, dim=-1)
                h_t_query = torch.matmul(score, h_t_pair)
                hq = self.score_linear2(h_t_query)
                hs = self.score_linear1(sents_con)
                p_score = hs + hq + 0.5
                p_score = p_score.squeeze()
                all_scores[batch_no][j][:p_score.shape[0]] = p_score
                all_scores_mask[batch_no][j][:p_score.shape[0]] = 1

        # evidence sentences
        select_sen = torch.gt(all_scores, 0.5).nonzero()
        select_sen_mask = torch.zeros(batch_size, h_t_num, input_sent.shape[1]).cuda()
        for (batch_no, h_t_no, sent_no) in select_sen.unbind(dim=0):
            sent_start = sent_indexs[batch_no][sent_no]
            sent_end = sent_indexs[batch_no][sent_no + 1]
            select_sen_mask[batch_no][h_t_no][sent_start:sent_end] = 1

        return all_scores[:,:,:max_sent_len], all_scores_mask[:,:,:max_sent_len], select_sen, select_sen_mask



    def forward(self, context_idxs, pos, context_ner, input_lengths, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, context_seg, mention_node_position, entity_position,
                mention_node_sent_num, all_node_num, entity_num_list, sdp_pos, sdp_num_list):
        """
        :param context_idxs: Token IDs
        :param pos: coref pos IDs
        :param context_ner: NER tag IDs
        :param h_mapping: Head
        :param t_mapping: Tail
        :param relation_mask: There are multiple relations for each instance so we need a mask in a batch
        :param dis_h_2_t: distance for head
        :param dis_t_2_h: distance for tail
        :param context_seg: mask for different sentences in a document
        :param mention_node_position: Mention node position
        :param entity_position: Entity node position
        :param mention_node_sent_num: number of mention nodes in each sentences of a document
        :param all_node_num: the number of nodes  (mention, entity, MDP) in a document
        :param entity_num_list: the number of entity nodes in each document
        :param sdp_pos: MDP node position
        :param sdp_num_list: the number of MDP node in each document
        :return:
        """

        '''===========STEP1: Encode the document============='''
        context_word = self.word_emb(context_idxs)
        all_scores, all_scores_mask, select_sen, select_sen_mask = self.select_sentences(context_word, context_seg, input_lengths, h_mapping, t_mapping)

        sent_emb = torch.cat([context_word, self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1)
        docs_rep, sents_rep = self.doc_encoder(sent_emb, context_seg)

        max_doc_len = docs_rep.shape[1]
        context_output = self.dropout_rate(torch.relu(self.linear_re(docs_rep)))


        '''===========STEP2: Extract all node reps of a document graph============='''
        '''extract Mention node representations'''
        mention_num_list = torch.sum(mention_node_sent_num, dim=1).long().tolist()
        max_mention_num = max(mention_num_list)
        mentions_rep = torch.bmm(mention_node_position[:, :max_mention_num, :max_doc_len], context_output) # mentions rep
        '''extract MDP(meta dependency paths) node representations'''
        sdp_num_list = sdp_num_list.long().tolist()
        max_sdp_num = max(sdp_num_list)
        sdp_rep = torch.bmm(sdp_pos[:,:max_sdp_num, :max_doc_len], context_output)
        '''extract Entity node representations'''
        entity_rep = torch.bmm(entity_position[:,:,:max_doc_len], context_output)
        '''concatenate all nodes of an instance'''
        gcn_inputs = []
        all_node_num_batch = []
        for batch_no, (m_n, e_n, s_n) in enumerate(zip(mention_num_list, entity_num_list.long().tolist(), sdp_num_list)):
            m_rep = mentions_rep[batch_no][:m_n]
            e_rep = entity_rep[batch_no][:e_n]
            s_rep = sdp_rep[batch_no][:s_n]
            gcn_inputs.append(torch.cat((m_rep, e_rep, s_rep),dim=0))
            node_num = m_n + e_n + s_n
            all_node_num_batch.append(node_num)

        gcn_inputs = pad_sequence(gcn_inputs).permute(1, 0, 2)
        output = gcn_inputs

        '''===========STEP3: Induce the Latent Structure============='''
        if self.use_reasoning_block:
            for i in range(len(self.reasoner)):
                output = self.reasoner[i](output)

        elif self.use_struct_att:
            gcn_inputs, _ = self.structInduction(gcn_inputs)
            max_all_node_num = torch.max(all_node_num).item()
            assert (gcn_inputs.shape[1] == max_all_node_num)

        mention_node_position = mention_node_position.permute(0, 2, 1)
        output = torch.bmm(mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num])
        context_output = torch.add(context_output, output)

        h_mapping = h_mapping.mul(select_sen_mask)
        t_mapping = t_mapping.mul(select_sen_mask)
        start_re_output = torch.matmul(h_mapping[:, :, :max_doc_len], context_output) # aggregation
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output) # aggregation

        s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

        re_rep = self.dropout_rate(self.relu(self.bili(s_rep, t_rep)))

        re_rep = self.self_att(re_rep, re_rep, relation_mask)
        return self.linear_output(re_rep), all_scores

