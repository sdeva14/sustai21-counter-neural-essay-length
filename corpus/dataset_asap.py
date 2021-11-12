import numpy as np
import pandas as pd

import utils
from utils import LONG, FLOAT

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

from corpus.dataset_base import Dataset_Base

# class Dataset_ASAP(Dataset):
class Dataset_ASAP(Dataset_Base):
	def __init__(self, data_id, config, pad_id):
		super().__init__(data_id, config, pad_id)
		# # train_data_id = {'x_data': x_id_train, 'y_label': y_train, 'origin_score': score_train}

		# # declared in base class
		# self.x_data = data_id["x_data"]
		# self.y_label = data_id["y_label"]

		# self.max_num_sents = config.max_num_sents
		# self.max_len_sent = config.max_len_sent
		# self.max_len_doc = config.max_len_doc
		# self.pad_level = config.pad_level

		# self.use_gpu = config.use_gpu

		self.origin_score = data_id["origin_score"]


		return

	#
	def __len__(self):
		return len(self.y_label)

	#
	def __getitem__(self, idx):
		cur_x = self.x_data[idx]
		cur_y = self.y_label[idx]
		tid = self.tid[idx]
		cur_origin_score = self.origin_score[idx]

		vec_text_input = None
		mask_input = None
		if self.pad_level == "sent" or self.pad_level == "sentence":
			vec_text_input, mask_input, seq_lens = self._pad_sent_level(cur_x)
		else:
			vec_text_input, mask_input, len_seq, len_sents = self._pad_doc_level(cur_x)
		# vec_text_input = utils.cast_type(vec_text_input, LONG, self.use_gpu)

		label_y = torch.FloatTensor([cur_y])
		# label_y = utils.cast_type(label_y, FLOAT, self.use_gpu)


		return vec_text_input, label_y, mask_input, len_seq, len_sents, tid, cur_origin_score


	# ######
	# def _pad_sent_level(self, doc_x):
	#     """ padding for sentence level """

	#     vec_text_input = torch.zeros(self.max_num_sents, self.max_len_sent, dtype=torch.int64)

	#     for ind_sent in range(len(doc_x)):
	#         non_padded_sent = torch.LongTensor(doc_x[ind_sent])

	#         pad_len = self.max_len_sent - len(non_padded_sent)
	#         padded_sent = F.pad(non_padded_sent, pad=(0, pad_len), mode='constant', value=0)
	#         vec_text_input[ind_sent, 0:len(padded_sent)] = padded_sent

	#     #
	#     cur_seq_len = 0
	#     for cur_sent in doc_x:
	#         cur_seq_len = cur_seq_len + len(cur_sent)

	#     return vec_text_input, cur_seq_len

	# #
	# def _pad_doc_level(self, doc_x):
	#     """ padding for document level """

	#     flat_sents = [item for sublist in doc_x for item in sublist]
	#     non_padded = torch.LongTensor(flat_sents)

	#     vec_text_input = torch.zeros(self.max_len_doc, dtype=torch.int64)
	#     pad_len = self.max_len_doc - len(non_padded)
	#     vec_text_input = F.pad(non_padded, pad=(0, pad_len), mode='constant', value=0)

	#     cur_seq_len = 0
	#     for cur_sent in doc_x:
	#         cur_seq_len = cur_seq_len + len(cur_sent)


	#     return vec_text_input, cur_seq_len

