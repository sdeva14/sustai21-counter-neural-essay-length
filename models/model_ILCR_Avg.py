# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import numpy as np
# import torch.distributions.normal as normal
import logging

import w2vEmbReader

from models.encoders.encoder_Coh import Encoder_Coh

import models.model_base
import utils
from utils import FLOAT

from scipy.stats import entropy
from math import log, e


class Coh_Model_ILCR_Avg(models.model_base.BaseModel):
    def __init__(self, config, corpus_target, embReader):
        # super(Coh_Model_ILCR_Simple, self).__init__(config)
        super().__init__(config)

        ####
        # init parameters
        self.corpus_target = config.corpus_target
        self.max_num_sents = config.max_num_sents  # document length, in terms of the number of sentences
        self.max_len_sent = config.max_len_sent  # sentence length, in terms of words
        self.max_len_doc = config.max_len_doc  # document length, in terms of words
        self.avg_len_doc = config.avg_len_doc   

        self.batch_size = config.batch_size


        self.vocab = corpus_target.vocab  # word2id
        self.rev_vocab = corpus_target.rev_vocab  # id2word
        self.pad_id = corpus_target.pad_id
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

        ########
        self.encoder_coh = Encoder_Coh(config, embReader)

        self.sim_cosine = torch.nn.CosineSimilarity(dim=2)

        #
        fc_in_size = self.encoder_coh.encoder_out_size
        # fc_in_size = self.encoder_coh.encoder_out_size + 1
        linear_1_out = fc_in_size // 2
        linear_2_out = linear_1_out // 2

        self.linear_1 = nn.Linear(fc_in_size, linear_1_out)

        self.bn1 = nn.BatchNorm1d(num_features=linear_1_out)
        #nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_normal_(self.linear_1.weight)

        self.linear_2 = nn.Linear(linear_1_out, linear_2_out)
        #nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_normal_(self.linear_2.weight)
        self.bn2 = nn.BatchNorm1d(num_features=linear_2_out)

        self.linear_out = nn.Linear(linear_2_out, self.output_size)
        if corpus_target.output_bias is not None:  # bias
            init_mean_val = np.expand_dims(corpus_target.output_bias, axis=1)
            bias_val = (np.log(init_mean_val) - np.log(1 - init_mean_val))
            self.linear_out.bias.data = torch.from_numpy(bias_val).type(torch.FloatTensor)
        #nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.xavier_normal_(self.linear_out.weight)

        #
        self.selu = nn.SELU()
        self.elu = nn.ELU()
        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.softmax = nn.Softmax(dim=1)

        return
    # end __init__

    # sort multi-dim array by 2d vs 2d dim (e.g., skip batch dim with higher than 2d indices)
    def sort_hid(self, input_unsorted, ind_len_sorted):
        squeezed = input_unsorted.squeeze(0)
        input_sorted = squeezed[ind_len_sorted]
        unsqueezed = input_sorted.unsqueeze(0)

        return unsqueezed
    # end sort_2nd_dim

    #
    def forward(self, text_inputs, mask_input, len_seq, len_sents, tid, mode=""):

        # #
        if self.pad_level == "sent" or self.pad_level == "sentence":
            text_inputs = text_inputs.view(text_inputs.size(0), text_inputs.size(1)*text_inputs.size(2))

        #
        encoder_out = self.encoder_coh(text_inputs, mask_input, len_seq)

        ## averaging RNN output by their length
        len_seq = utils.cast_type(len_seq, FLOAT, self.use_gpu)
        ilc_vec = torch.div(torch.sum(encoder_out, dim=1), self.avg_len_doc)  # (batch_size, rnn_cell_size)

        #### Fully Connected
        fc_out = self.linear_1(ilc_vec)
        fc_out = self.leak_relu(fc_out)
        fc_out = self.dropout_layer(fc_out)

        fc_out = self.linear_2(fc_out)
        fc_out = self.leak_relu(fc_out)
        fc_out = self.dropout_layer(fc_out)

        fc_out = self.linear_out(fc_out)
        
        if self.corpus_target.lower() == "asap":
            fc_out = self.sigmoid(fc_out)


        return fc_out
    # end forward
