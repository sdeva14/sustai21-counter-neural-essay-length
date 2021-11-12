# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
import numpy as np
import torch.nn.functional as F

import utils
from utils import INT, LONG, FLOAT # macro for 0, 1, 2
import math

import logging

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.flush_valid = False
        self.config = config
        self.kl_w = 0.0  # it does not need now?

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return utils.cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.use_gpu)

    def forward(self, *input):
        raise NotImplementedError

    def Linear(self, in_features, out_features, bias=True):
        m = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(m.weight)
        if bias:
            nn.init.constant_(m.bias, 0.)
        return m

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)

    def _gather_last_out(self, rnn_outs, lens):
        """
        :param rnn_outs: batch_size x T_len x dimension
        :param lens: [a list of lens]
        :return: batch_size x dimension
        """
        time_dimension = 1
        len_vars = self.np2var(np.array(lens), LONG)
        len_vars = len_vars.view(-1, 1).expand(len(lens), rnn_outs.size(2)).unsqueeze(1)
        slices = rnn_outs.gather(time_dimension, len_vars-1)
        return slices.squeeze(time_dimension)

    def _remove_padding(self, feats, words):
        """"
        :param feats: batch_size x num_words x feats
        :param words: batch_size x num_words
        :return: the same input without padding
        """
        if feats is None:
            return None, None

        batch_size = words.size(0)
        valid_mask = torch.sign(words).float()
        batch_lens = torch.sum(valid_mask, dim=1)
        max_word_num = torch.max(batch_lens)
        padded_lens = (max_word_num - batch_lens).cpu().data.numpy()
        valid_words = []
        valid_feats = []

        for b_id in range(batch_size):
            valid_idxs = valid_mask[b_id].nonzero().view(-1)
            valid_row_words = torch.index_select(words[b_id], 0, valid_idxs)
            valid_row_feat = torch.index_select(feats[b_id], 0, valid_idxs)

            padded_len = int(padded_lens[b_id])
            valid_row_words = F.pad(valid_row_words, (0, padded_len))
            valid_row_feat = F.pad(valid_row_feat, (0, 0, 0, padded_len))

            valid_words.append(valid_row_words.unsqueeze(0))
            valid_feats.append(valid_row_feat.unsqueeze(0))

        feats = torch.cat(valid_feats, dim=0)
        words = torch.cat(valid_words, dim=0)
        return feats, words

    def get_optimizer(self, config):
        logger = logging.getLogger() 
        if config.op == 'adam':
            logger.info("Optimizer: adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr, eps=config.eps, weight_decay=config.lr_decay)
                                           # self.parameters()), lr=config.init_lr, eps=config.eps, amsgrad=True)
        if config.op == 'adamw':
            logger.info("Optimizer: AdamW")
            return torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr, eps=config.eps, weight_decay=config.lr_decay)
        elif config.op == 'sgd':
            logger.info("Optimizer: SGD")
            return torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'asgd':
            logger.info("Optimizer: ASGD")
            return torch.optim.ASGD(self.parameters(), lr=config.init_lr)

        elif config.op == 'rmsprop':
            logger.info("Optimizer: RMSProp")
            return torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
        # elif config.op.lower() == 'adamw':
        #     logger.info("Optimizer: AdamW")
        #     from pytorch_transformers import AdamW
        #     return AdamW(filter(lambda p: p.requires_grad,
        #                                    self.parameters()), lr=config.init_lr, eps=config.eps, weight_decay=config.lr_decay)

    def gelu(self, x):
        """ Implementation of the gelu activation function.
            XLNet is using OpenAI GPT's gelu (not exactly the same as BERT)
            Also see https://arxiv.org/abs/1606.08415
        """
        cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * cdf

