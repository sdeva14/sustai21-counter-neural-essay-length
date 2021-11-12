# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

import models.drnn as drnn


class Encoder_DRNN(nn.Module):
    """ encoders class """

    def __init__(self, config, x_embed):
        super().__init__()

        self.num_layers_drnn = config.drnn_layer
        self.x_embed = x_embed.x_embed
        self.model = drnn.DRNN(n_input=x_embed.embedding_dim,
                               n_hidden=config.rnn_cell_size,
                               n_layers=self.num_layers_drnn,
                               dropout=config.dropout,
                               cell_type='GRU',
                               batch_first=True)
        self.encoder_out_size = config.rnn_cell_size

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
        mask = torch.sign(text_inputs)  # (batch_size, max_doc_len)
        len_seq_sent = mask.sum(dim=1)
        len_seq_sent_sorted, ind_len_sorted = torch.sort(len_seq_sent,
                                                         descending=True)  # ind_len_sorted: (batch_size, num_sents)
        #
        sent_x_input_sorted = x_input[ind_len_sorted]

        sent_lstm_out, _ = self.model(sent_x_input_sorted)  # out: (batch_size, len_sent, cell_size)

        # revert to origin order
        _, ind_origin = torch.sort(ind_len_sorted)
        sent_lstm_out = sent_lstm_out[ind_origin]

        # masking
        cur_sents_mask = torch.sign(len_seq_sent)
        cur_sents_mask = cur_sents_mask.view(-1, 1, 1)  # (batch_size, num_sents)
        encoder_out = sent_lstm_out * cur_sents_mask.expand_as(sent_lstm_out).float()  # masking

        return encoder_out

        return

# end class
