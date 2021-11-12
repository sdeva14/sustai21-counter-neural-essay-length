# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import logging

import w2vEmbReader
from corpus.corpus_base import PAD, BOS, EOS

import models.model_base
import models.drnn as drnn
import utils
from utils import INT, FLOAT, LONG
from copy import deepcopy

from models.reg_lstm.weight_drop import WeightDrop
from models.reg_lstm.embed_regularize import embedded_dropout
from models.reg_lstm.locked_dropout import LockedDropout

from models.encoders.encoder_rnn import Encoder_RNN
from models.encoders.encoder_drnn import Encoder_DRNN
from models.encoders.encoder_bert import Encoder_BERT
from models.encoders.encoder_reglstm import Encoder_RegLSTM
# from models.encoders.encoder_transformer import Encoder_Transfomer
from models.encoders.StructuredAttention import StructuredAttention
from models.encoders.encoder_xlnet import Encoder_XLNet


class Encoder_Coh(nn.Module):
    """ encoders class """

    def __init__(self, config, x_embed, input_size=None, hid_size=None):
        super(Encoder_Coh, self).__init__()

        self.encoder_type = config.encoder_type.lower()
        self.use_gpu = config.use_gpu

        self.rnn_cell_type = config.rnn_cell_type  # can be changed according to encoder_type
        logger = logging.getLogger()

        #
        self.encoder = None
        if self.encoder_type == "lstm" or self.encoder_type == "gru":
            logger.info("Encoder: RNN")
            self.encoder = Encoder_RNN(config, x_embed, input_size, hid_size)
        elif self.encoder_type == "drnn":
            logger.info("Encoder: DRNN")
            self.encoder = Encoder_DRNN(config, x_embed)
        elif self.encoder_type == "bert":
            logger.info("Encoder: BERT")
            self.encoder = Encoder_BERT(config, x_embed)
        elif self.encoder_type == "reg_lstm":
            logger.info("Encoder: Reg_LSTM")
            self.encoder = Encoder_RegLSTM(config, x_embed)
        elif self.encoder_type == "transf":
            logger.info("Encoder: Transf")
            self.encoder = Encoder_Transfomer(config, x_embed)
        elif self.encoder_type == "stru_attn":
            logger.info("Encoder: Stru_Attn")
            self.encoder = StructuredAttention(config, x_embed)
        elif self.encoder_type == "xlnet":
            logger.info("Encoder: XLNet")
            self.encoder = Encoder_XLNet(config, x_embed)

        self.encoder_out_size = self.encoder.encoder_out_size

        self.beta_ema = config.beta_ema
        if self.beta_ema>0:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if self.use_gpu:
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        return
    # end __init__

    # sort multi-dim array by 2d vs 2d dim (e.g., skip batch dim with higher than 2d indices)
    def sort_hid(self, input_unsorted, ind_len_sorted):
        squeezed = input_unsorted.squeeze(0)
        input_sorted = squeezed[ind_len_sorted]
        unsqueezed = input_sorted.unsqueeze(0)

        return unsqueezed
    # end sort_2nd_dim

    # temporal averaging proposed in ICLR18
    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1-self.beta_ema)*p.data) 

    #
    def forward(self, text_inputs, mask_input, len_seq, mode=""):
        encoder_out = self.encoder(text_inputs, mask_input, len_seq, mode)

        return encoder_out

    #
    def forward_skip(self, x_input, len_seq, mode=""):  # skip embedding part when embedded input is given
        encoder_out = self.encoder.forward_skip(x_input, len_seq, mode)

        return encoder_out

