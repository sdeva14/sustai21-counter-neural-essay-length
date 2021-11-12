# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

from copy import deepcopy

from models.reg_lstm.weight_drop import WeightDrop
from models.reg_lstm.embed_regularize import embedded_dropout
from models.reg_lstm.locked_dropout import LockedDropout

class Encoder_RegLSTM(nn.Module):
    """ encoders class """

    def __init__(self, config, x_embed):
        super().__init__()

        self.num_layers_rnn = 1
        self.x_embed = x_embed.x_embed

        self.wdrop = config.wdrop
        self.dropoute = config.dropoute
        self.encoder_out_size = config.rnn_cell_size
        self.rnn_cell_type = config.rnn_cell_type

        self.training = True

        import warnings
        warnings.filterwarnings("ignore")

        self.model = None
        if self.rnn_cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=x_embed.embedding_dim,
                               hidden_size=config.rnn_cell_size,
                               num_layers=self.num_layers_rnn,
                               bidirectional=False,
                               dropout=config.dropout,
                               batch_first=True,
                               bias=True)
            self.model = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=self.wdrop)

        elif self.rnn_cell_type.lower() == "gru":
            self.rnn = nn.GRU(input_size=x_embed.embedding_dim,
                              hidden_size=config.rnn_cell_size,
                              num_layers=self.num_layers_rnn,
                              bidirectional=False,
                              dropout=config.dropout,
                              batch_first=True,
                              bias=True)
            self.model = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=self.wdrop)

        elif self.rnn_cell_type.lower() == "qrnn":
            from torchqrnn import QRNNLayer
            self.model = QRNNLayer(input_size=x_embed.embedding_dim,
                                   hidden_size=config.rnn_cell_size,
                                   save_prev_x=True,
                                   zoneout=0,
                                   window=1,
                                   output_gate=True,
                                   use_cuda=config.use_gpu)
            self.model.linear = WeightDrop(self.model.linear, ['weight'], dropout=self.wdrop)
            # self.encoders.reset()

        self.lockdrop = LockedDropout()
        self.dropouti = 0.1

        # temporal averaging
        self.beta_ema = config.beta_ema
        if self.beta_ema > 0:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if config.use_gpu:
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

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
        # embedded_drouput
        x_input = embedded_dropout(self.x_embed, text_inputs, dropout=self.dropoute if self.training else 0)
        # x_input = self.lockdrop(x_input, self.dropouti)  # not sure it really need

        # mask = torch.sign(text_inputs)  # (batch_size, max_doc_len)
        # len_seq_sent = mask.sum(dim=1)
        mask = mask_input.view(text_inputs.shape)
        len_seq_sent_sorted, ind_len_sorted = torch.sort(len_seq,
                                                         descending=True)  # ind_len_sorted: (batch_size, num_sents)

        #
        sent_x_input_sorted = x_input[ind_len_sorted]

        if self.rnn_cell_type.lower() == "qrnn":  # when batch_first is not supported
           sent_x_input_sorted = sent_x_input_sorted.transpose(0, 1)

        # self.rnn.flatten_parameters()
        sent_lstm_out, _ = self.model(sent_x_input_sorted)  # out: (batch_size, len_sent, cell_size)
        sent_lstm_out = self.lockdrop(sent_lstm_out, self.dropouti)  # not sure it really need

        if self.rnn_cell_type.lower() == "qrnn":  # when batch_first is not supported
            sent_lstm_out = sent_lstm_out.transpose(1, 0)

        # revert to origin order
        _, ind_origin = torch.sort(ind_len_sorted)
        sent_lstm_out = sent_lstm_out[ind_origin]

        # masking
        # cur_sents_mask = torch.sign(len_seq_sent)
        # cur_sents_mask = cur_sents_mask.view(-1, 1, 1)  # (batch_size, num_sents)
        # encoder_out = sent_lstm_out * cur_sents_mask.expand_as(sent_lstm_out).float()  # masking
        encoder_out = sent_lstm_out * mask.unsqueeze(2)

        return encoder_out

# end class
