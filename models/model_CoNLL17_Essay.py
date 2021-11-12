# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import numpy as np
import logging

import w2vEmbReader

from models.encoders.encoder_Coh import Encoder_Coh

import models.model_base
import utils
from utils import FLOAT


class Model_CoNLL17_Essay(models.model_base.BaseModel):
    """ class for CoNLL17 implementation
        Title: A Neural Approach to Automated Essay Scoring
        Ref: https://www.aclweb.org/anthology/D16-1193/
    """

    def __init__(self, config, corpus_target, embReader):
        super().__init__(config)

        ####
        self.corpus_target = config.corpus_target
        self.target_model = config.target_model.lower()

        # init parameters
        self.max_num_sents = config.max_num_sents  # document length, in terms of the number of sentences
        self.max_len_sent = config.max_len_sent  # sentence length, in terms of words
        self.max_len_doc = config.max_len_doc  # document length, in terms of words
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
        #
        self.encoder_base = Encoder_Main(config, embReader)

        #
        self.conv_size = 5
        self.conv = nn.Conv1d(in_channels=1,  # conv size is 5 according to the original impl
                      out_channels=100,
                      kernel_size=self.conv_size,
                      stride=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      bias=True)
        self.conv_output_size = 100  # according to original paer
        
        # original impl: extend to 100 dim by kernel, then avg pool to 1 for each kernel dim (following original impl)
        self.size_avg_pool = 1
        self.avg_adapt_pool1 = nn.AdaptiveAvgPool1d(self.size_avg_pool)

        #
        # fc_in_size = self.encoder_coh.encoder_out_size
        fc_in_size = self.conv_output_size
        linear_1_out = fc_in_size // 2
        linear_2_out = linear_1_out // 2

        # implement attention by linear (general version)
        self.attn = nn.Linear(self.max_len_doc * self.conv_output_size, self.max_len_doc, bias=True)
        #nn.init.xavier_uniform_(self.attn.weight)
        nn.init.xavier_normal_(self.attn.weight)

        # implement attention by parameter (bahdanau style)
        self.word_weight = nn.Parameter(torch.Tensor(self.conv_output_size, self.conv_output_size))
        self.word_bias = nn.Parameter(torch.zeros(self.conv_output_size))
        self.context_weight = nn.Parameter(torch.zeros(self.conv_output_size))
        nn.init.xavier_normal_(self.word_weight)

        
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
        #
        if self.pad_level == "sent" or self.pad_level == "sentence":
            text_inputs = text_inputs.view(text_inputs.shape[0], self.max_num_sents*self.max_len_sent)
        mask = mask_input.view(text_inputs.shape)

        #
        encoder_out = self.encoder_base(text_inputs, mask_input, len_seq)

        # applying conv1d after rnn
        avg_pooled = torch.zeros(text_inputs.shape[0], text_inputs.shape[1], self.conv_output_size)
        avg_pooled = utils.cast_type(avg_pooled, FLOAT, self.use_gpu)
        for cur_batch, cur_tensor in enumerate(encoder_out):
            ## Actual length version
            if self.target_model == "conll17_al":
                cur_seq_len = int(len_seq[cur_batch])
                cur_tensor = cur_tensor.unsqueeze(0)
                crop_tensor = cur_tensor.narrow(1, 0, cur_seq_len)
                crop_tensor = crop_tensor.transpose(1, 0)
                cur_tensor = crop_tensor
            ## published version: do not consider actual length
            else:
                cur_tensor = cur_tensor.unsqueeze(1)

            # applying conv
            cur_tensor = self.conv(cur_tensor)  
            cur_tensor = self.leak_relu(cur_tensor)
            cur_tensor = self.dropout_layer(cur_tensor)
            # cur_tensor = self.avg_pool_1d(cur_tensor)
            cur_tensor  = self.avg_adapt_pool1(cur_tensor)
            cur_tensor = cur_tensor.view(cur_tensor.shape[0], self.conv_output_size)
            avg_pooled[cur_batch, :cur_tensor.shape[0], :] = cur_tensor

        len_seq = utils.cast_type(len_seq, FLOAT, self.use_gpu)

        ## implement attention by parameters
        context_weight = self.context_weight.unsqueeze(1)
        context_weight = context_weight.expand(text_inputs.shape[0], self.conv_output_size, 1)
        attn_weight = torch.bmm(avg_pooled, context_weight).squeeze(2)
        attn_weight = self.tanh(attn_weight)
        attn_weight = self.softmax(attn_weight)
        # attention applied
        attn_vec = torch.bmm(avg_pooled.transpose(1, 2), attn_weight.unsqueeze(2))
        
        ilc_vec = attn_vec.squeeze(2)

        ## implement attention by linear
        #attn_vec = self.attn(encoder_out.view(self.batch_size, -1)).unsqueeze(2)
        #attn_vec = self.softmax(attn_vec)
        #ilc_vec_attn = torch.bmm(encoder_out.transpose(1, 2), attn_vec).squeeze(2)

        ## FC

        # fully connected stage
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
