# -*- coding: utf-8 -*-
# implementation of "SKIPFLOW: Incorporating Neural Coherence Features for End-to-End Automatic Text Scoring" in AAAI18

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import torch.distributions.normal as normal
import math
import logging

import w2vEmbReader
from corpus.corpus_base import PAD, BOS, EOS

import models.model_base
import utils
from utils import INT, FLOAT, LONG


class Neural_Tensor_Layer(nn.Module):
    """ subclass in the AAAI18 implemenation, named neural tensor feature in the paper"""
    #
    def __init__(self, batch_size, rnn_cell_size, dim_tensor_feat, skip_num, use_gpu, use_parallel):
        super(Neural_Tensor_Layer, self).__init__()

        self.rnn_cell_size = rnn_cell_size
        self.dim_tensor_feat = dim_tensor_feat
        self.batch_size = batch_size
        self.skip_num = skip_num
        self.use_gpu = use_gpu

        self.M = torch.FloatTensor(self.dim_tensor_feat, self.rnn_cell_size, self.rnn_cell_size).uniform_(-2, 2)
        self.M = utils.cast_type(self.M, FLOAT, self.use_gpu)
        self.V = torch.FloatTensor(2 * self.rnn_cell_size, self.dim_tensor_feat).uniform_(-2, 2)
        self.V = utils.cast_type(self.V, FLOAT, self.use_gpu)
        
        if self.use_gpu:
            self.V = self.V.to(torch.device("cuda:0"))

        bias_batch = self.batch_size 
        if self.use_gpu and use_parallel:
            bias_batch = int(bias_batch / torch.cuda.device_count())

        self.bias = torch.zeros(bias_batch, self.skip_num, self.dim_tensor_feat)
        self.bias = utils.cast_type(self.bias, FLOAT, self.use_gpu)

        self.M.requires_grad = True
        self.V.requires_grad = True
        self.bias.requires_grad = True

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        return
    # end def __init__

    #
    def forward(self, state_pair):  # state_pair: pair of (batch_size, hidden_size)
        # states = torch.cat((state_pair[0], state_pair[1]), dim=1)

        state_a = [p[0] for p in state_pair]
        state_a = torch.stack(state_a).transpose(1,0)
        state_b = [p[1] for p in state_pair]
        state_b = torch.stack(state_b).transpose(1,0)

        states = [torch.cat((p[0], p[1]), dim=1) for p in state_pair]  # list of concantened states (batch_size, 2 * rnn_size)
        states = torch.stack(states).transpose(1,0)  # (batch_size, num_skipping, rnn_size)

        v_term = torch.matmul(states, self.V)
        billinear_products = []
        for i in range(self.dim_tensor_feat):
            product_term = torch.matmul(state_a, self.M[i])
            bi_product = torch.matmul(state_b, product_term.transpose(2,1))
            bi_product_term = bi_product.sum(dim=1)
            billinear_products.append(bi_product_term)

        tensor_feat_output = torch.stack(billinear_products).permute(1, 2, 0)  #.transpose(1,0)
        bias = self.bias[:tensor_feat_output.shape[0], :, :]

        tensor_feat_output = tensor_feat_output + v_term + bias
        tensor_feat_output = self.tanh(tensor_feat_output)
        tensor_feat_output = self.sigmoid(tensor_feat_output)  # (batch_size, skip_num, dim_tensor_feet)

        return tensor_feat_output
    # end def forward

#####################################################################
#####################################################################

class Coh_Model_AAAI18(models.model_base.BaseModel):
    """ class for AAAI18 implementation
        Title: SKIPFLOW: Incorporating Neural Coherence Features for End-to-End Automatic Text Scoring
        Ref: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16431/16161
    """

    def __init__(self, config, corpus_target, embReader):
        super(Coh_Model_AAAI18, self).__init__(config)

        ####
        # init parameters
        self.corpus_target = config.corpus_target
        self.target_model = config.target_model.lower()

        self.max_num_sents = config.max_num_sents  # document length, in terms of the number of sentences
        self.max_len_sent = config.max_len_sent  # sentence length, in terms of words
        self.max_len_doc = config.max_len_doc  # document length, in terms of words
        self.batch_size = config.batch_size

        self.vocab = corpus_target.vocab  # word2id
        self.rev_vocab = corpus_target.rev_vocab  # id2word
        self.vocab_size = len(self.vocab)
        self.pad_id = self.rev_vocab[PAD]
        self.bos_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.num_special_vocab = corpus_target.num_special_vocab

        self.embed_size = config.embed_size
        self.dropout_rate = config.dropout
        self.rnn_cell_size = config.rnn_cell_size
        self.path_pretrained_emb = config.path_pretrained_emb
        self.num_layers = 1
        self.output_size = config.output_size  # the number of final output class
        self.pad_level = config.pad_level

        self.use_gpu = config.use_gpu

        if not hasattr(config, "freeze_step"):
            config.freeze_step = 5000

        self.skip_start = config.skip_start
        self.skip_jump = config.skip_jump
        self.dim_tensor_feat = config.dim_tensor_feat

        ####
        ## model architecture

        # embeding for input (1st-stage in the paper)
        self.x_embed = embReader.get_embed_layer()

        # LSTM after embedding (2nd-stage in the paper)
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.lstm = nn.LSTM(input_size=self.x_embed.embedding_dim,
                            hidden_size=self.rnn_cell_size,
                            num_layers=self.num_layers,
                            bidirectional=False,
                            dropout=self.dropout_rate,
                            batch_first=True,
                            bias=True)
        self.lstm.apply(self._init_weights)

        skip_num = self.max_len_doc // self.skip_jump
        self.model_tensor_feature = Neural_Tensor_Layer(config.batch_size, config.rnn_cell_size, self.dim_tensor_feat, skip_num, self.use_gpu, config.use_parallel)

        self.linear_128_d = nn.Linear(skip_num * self.dim_tensor_feat + self.rnn_cell_size, 128)
        nn.init.xavier_uniform_(self.linear_128_d.weight)

        self.linear_256 = nn.Linear(skip_num * self.dim_tensor_feat + self.rnn_cell_size, 256)

        nn.init.xavier_uniform_(self.linear_256.weight)
        self.linear_128 = nn.Linear(256, 128)
        nn.init.xavier_uniform_(self.linear_128.weight)
        self.linear_64 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.linear_64.weight)
        self.linear_out = nn.Linear(64, self.output_size)
        if corpus_target.output_bias is not None:  # bias
            init_mean_val = np.expand_dims(corpus_target.output_bias, axis=1)
            bias_val = (np.log(init_mean_val) - np.log(1 - init_mean_val))
            self.linear_out.bias.data = torch.from_numpy(bias_val).type(torch.FloatTensor)
        nn.init.xavier_uniform_(self.linear_out.weight)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leak_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax()

        return
    # end def __init__

    #
    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    # end def _init_weights

    # sort multi-dim array by 2d vs 2d dim (e.g., skip batch dim with higher than 2d indices)
    def sort_hid(self, input_unsorted, ind_len_sorted):
        squeezed = input_unsorted.squeeze(0)
        input_sorted = squeezed[ind_len_sorted]
        unsqueezed = input_sorted.unsqueeze(0)

        return unsqueezed
    # end sort_2nd_dim

    # get pair states considering actual length (al)
    def _get_states_pair_al(self, len_seq, pairs, sent_lstm_out):
        states_pairs = []

        # handle for each document with their actual length
        ## first, get proper index with actual length
        p1s_raw = [i[0] for i in pairs]  # raw index only by max length
        p2s_raw = [i[1] for i in pairs]
        list_ind_1 = [cur_ind_1 % int(cur_len) for cur_len in len_seq for cur_ind_1 in p1s_raw]  # considering each len
        list_ind_2 = [cur_ind_2 % int(cur_len) for cur_len in len_seq for cur_ind_2 in p2s_raw]

        pairs_al = []
        for i in range(self.max_len_doc // self.skip_jump):
            col_1 = list_ind_1[i::len(pairs)]
            col_2 = list_ind_2[i::len(pairs)]
            cur_pair = (col_1, col_2)
            pairs_al.append(cur_pair)

        ## second, get states by pairs_al
        for cur_pair in pairs_al:

            # make a index for batch
            idx_1 = torch.LongTensor(cur_pair[0])
            idx_1 = utils.cast_type(idx_1, LONG, self.use_gpu)
            idx_1 = idx_1.view(-1, 1, 1).expand(-1, 1, self.rnn_cell_size)
            # batch select for states by index
            sent_out_1 = sent_lstm_out.gather(1, idx_1)
            sent_out_1 = sent_out_1.squeeze(1)

            idx_2 = torch.LongTensor(cur_pair[1])
            idx_2 = utils.cast_type(idx_2, LONG, self.use_gpu)
            idx_2 = idx_2.view(-1, 1, 1).expand(-1, 1, self.rnn_cell_size)
            sent_out_2 = sent_lstm_out.gather(1, idx_2)
            sent_out_2 = sent_out_2.squeeze(1)

            cur_state_pair = (sent_out_1, sent_out_2)
            states_pairs.append(cur_state_pair)


        return states_pairs

    #
    def forward(self, text_inputs, mask_input, len_seq, len_sents, mode=""):
        # it handles document-level padding for essay (stage 1)
        if self.pad_level=="sent" or self.pad_level=="sentence":
            text_inputs = text_inputs.view(self.batch_size, self.max_num_sents*self.max_len_sent)
        x_input = self.x_embed(text_inputs)  # 3dim, (batch_size, max_doc_len, embed_size)

        # get lstm output from embed input (stage2)
        mask = mask_input.view(text_inputs.shape)
        
        len_seq_sorted, ind_len_sorted = torch.sort(len_seq, descending=True)  # ind_len_sorted: (batch_size, num_sents)
        sent_x_input_sorted = x_input[ind_len_sorted]

        self.lstm.flatten_parameters()
        sent_lstm_out, _ = self.lstm(sent_x_input_sorted)  # out: (batch_size, len_sent, cell_size)

        # revert to origin order
        _, ind_origin = torch.sort(ind_len_sorted)
        sent_lstm_out = sent_lstm_out[ind_origin]

        # masking
        sent_lstm_out = sent_lstm_out * mask.unsqueeze(2)

        # temporal mean pooling (stage 3)
        len_seq = utils.cast_type(len_seq, FLOAT, self.use_gpu)
        temporal_mean_pooling = torch.div(torch.sum(sent_lstm_out, dim=1), len_seq.unsqueeze(1))  # (batch_size, rnn_cell_size)
        # temporal_mean_pooling = torch.div(torch.sum(sent_lstm_out, dim=1), self.rnn_cell_size)  # (batch_size, rnn_cell_size)
        temporal_mean_pooling = utils.cast_type(temporal_mean_pooling, FLOAT, self.use_gpu)

        # tensor layer feature (stage 4)
        # range_pair = [int(cur_len_seq) // self.skip_jump for cur_len_seq in len_seq]
        pairs = [((self.skip_start + i * self.skip_jump) % self.max_len_doc, (self.skip_start + i * self.skip_jump + self.skip_jump) % self.max_len_doc) for i in range(self.max_len_doc // self.skip_jump)]
        # pairs = [((self.skip_start + i * self.skip_jump) % self.max_len_doc, (self.skip_start + i * self.skip_jump + self.skip_jump) % self.max_len_doc) for i in range_pair]

        ## proper impl considering actual length of sequence
        if self.target_model == "aaai18_al":
            states_pairs = self._get_states_pair_al(len_seq, pairs, sent_lstm_out)
        ## published impl without considering actual length of sequence
        else:
            states_pairs = [(sent_lstm_out[:, p[0], :], sent_lstm_out[:, p[1], :]) for p in pairs]  # pair list of (batch_size, hidden_size)
        
        ##
        tensor_feat_output = self.model_tensor_feature(states_pairs)  # (batch_size, skip_num, dim_tensor_feat)

        # concat (temporal mean pooling) and (neural tensor feature)
        temporal_mean_pooling = temporal_mean_pooling.view(temporal_mean_pooling.shape[0], -1)
        tensor_feat_output = tensor_feat_output.view(tensor_feat_output.shape[0], -1)
        
        coh_layer = torch.cat((temporal_mean_pooling, tensor_feat_output), dim=1)

        # fully-connected hidden layer (stage 5)
        fc_out = self.linear_256(coh_layer)
        fc_out = self.leak_relu(fc_out)  # note that the author clearly mention that no dropout here
        fc_out = self.linear_128(fc_out)
        fc_out = self.leak_relu(fc_out)
        fc_out = self.linear_64(fc_out)
        fc_out = self.leak_relu(fc_out)

        #
        fc_out = self.linear_out(fc_out)
        if self.corpus_target.lower() == "asap":
            fc_out = self.sigmoid(fc_out)

        return fc_out
    # end forward
