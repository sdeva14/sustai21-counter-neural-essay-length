# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

import utils
from utils import FLOAT


class Encoder_RNN(nn.Module):
    """ encoders class """

    def __init__(self, config, emb_reader, input_size=None, hid_size=None):
        super().__init__()

        if input_size is not None:
            self.rnn_input_size = input_size
        else:
            self.rnn_input_size = emb_reader.embedding_dim
        if hid_size is not None:
            self.rnn_hid_size = hid_size
        else:
            self.rnn_hid_size = config.rnn_cell_size

        #if config.rnn_bidir:
        #    self.rnn_hid_size = self.rnn_hid_size // 2

        if config.encoder_type == "lstm":
            self.rnn_cell_type = "lstm"  # in this case, encoders type is equal to rnn_cell type
            self.num_layers_lstm = config.rnn_num_layer  # default 1
            self.x_embed = emb_reader.x_embed
            self.model = nn.LSTM(input_size=self.rnn_input_size,
                                 hidden_size=self.rnn_hid_size,
                                 num_layers=self.num_layers_lstm,
                                 bidirectional=config.rnn_bidir,
                                 dropout=config.dropout,
                                 batch_first=True,
                                 bias=True)

        elif config.encoder_type == "gru":
            self.rnn_cell_type = "gru"  # in this case, encoders type is equal to rnn_cell type
            self.num_layers_gru = config.rnn_num_layer  # default 1
            self.x_embed = emb_reader.x_embed
            self.model = nn.GRU(input_size=self.rnn_input_size,
                                hidden_size=self.rnn_hid_size,
                                num_layers=self.num_layers_gru,
                                bidirectional=config.rnn_bidir,
                                dropout=config.dropout,
                                batch_first=True,
                                bias=True)
        #    
        self.model.apply(self._init_weights)
        
        self.encoder_out_size = config.rnn_cell_size
        if config.rnn_bidir:
            self.encoder_out_size = self.encoder_out_size * 2

        self.tokenizer_type = config.tokenizer_type

        #if config.use_gpu:
        #    self.x_embed = utils.cast_type(self.x_embed, FLOAT, config.use_gpu)

        return
    # end init

    #
    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    # end def _init_weights

    #
    def forward(self, text_inputs, mask_input, len_seq, mode=""):

        x_input = self.x_embed(text_inputs)
        # mask = torch.sign(text_inputs)  # (batch_size, max_doc_len)
        mask = mask_input.view(text_inputs.shape)

        # len_seq_sent = mask.sum(dim=1)
        len_seq_sent_sorted, ind_len_sorted = torch.sort(len_seq,
                                                         descending=True)  # ind_len_sorted: (batch_size, num_sents)
        #
        sent_x_input_sorted = x_input[ind_len_sorted]
        self.model.flatten_parameters()  # error in pytorch 1.1.0 version with distributed api

        sent_lstm_out, _ = self.model(sent_x_input_sorted)  # out: (batch_size, len_sent, cell_size)

        # revert to origin order
        _, ind_origin = torch.sort(ind_len_sorted)
        sent_lstm_out = sent_lstm_out[ind_origin]

        # masking
        # cur_sents_mask = torch.sign(len_seq)
        # cur_sents_mask = cur_sents_mask.view(-1, 1, 1)  # (batch_size, num_sents)
        # encoder_out = sent_lstm_out * cur_sents_mask.expand_as(sent_lstm_out).float()  # masking
        encoder_out = sent_lstm_out * mask.unsqueeze(2)
        # encoder_out = sent_lstm_out

        return encoder_out
    # end forward

    #
    def forward_skip(self, x_input, mask, len_seq, mode=""):
        ''' skip embedding part when embedded input is given '''
        # mask = mask_input.view(x_input.shape)
        len_seq_sent_sorted, ind_len_sorted = torch.sort(len_seq,
                                                         descending=True)  # ind_len_sorted: (batch_size, num_sents)
        #
        sent_x_input_sorted = x_input[ind_len_sorted]
        # self.model.flatten_parameters()

        sent_lstm_out, _ = self.model(sent_x_input_sorted)  # out: (batch_size, len_sent, cell_size)

        # revert to origin order
        _, ind_origin = torch.sort(ind_len_sorted)
        encoder_out = sent_lstm_out[ind_origin]

        # masking
        # if self.tokenizer_type.startswith('word'):
        encoder_out = sent_lstm_out * mask.unsqueeze(2)  # with zero masking

        return encoder_out
    # end forward

# end class





