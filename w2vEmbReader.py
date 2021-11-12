# -*- coding: utf-8 -*-
import codecs
import logging
import numpy as np
import torch.nn as nn
import torch

import gensim

from corpus.corpus_base import PAD, BOS, EOS

logger = logging.getLogger()

# from pytorch_pretrained_bert import BertModel
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_transformers import BertConfig, BertTokenizer, BertModel
import utils
from utils import FLOAT

class W2VEmbReader:
	def __init__(self, config, corpus_target):
		logger.info('Loading embeddings from: ' + config.path_pretrained_emb)

		# Init parameters
		self.emb_path = config.path_pretrained_emb
		self.embed_size = config.embed_size
		if config.encoder_type =="transf":
			self.embed_size = config.d_model
		self.embedding_dim = None

		self.embeddings = {}
		self.emb_matrix = None

		self.x_embed = None  # embedding layer will be returned

		# load embedding
		self.pad_id = None
		if self.emb_path.startswith("bert-"):
			# Bert returns embedding layer itself, not matrix such as other pretrained embedding
			self.x_embed = self.load_pretrained_bert() # from bert library
			self.embedding_dim = 768
			self.pad_id = 0
		elif self.emb_path.startswith("xlnet-"):
			self.x_embed = None  # xlnet does not use additional pretrained embedding class
			self.embedding_dim = 768
			self.pad_id = 5
		else:
			# manual version
			self.vocab_size = len(corpus_target.vocab)
			self.vocab = corpus_target.vocab  # word2id
			self.rev_vocab = corpus_target.rev_vocab  # id2word
			# self.pad_id = self.rev_vocab[PAD]
			self.pad_id = corpus_target.pad_id

			self.num_special_vocab = corpus_target.num_special_vocab


			if not self.emb_path.lower().startswith("none") and len(self.emb_path) > 1:
				self.load_pretrained_file()  # from pretrained file
			self.emb_matrix = np.zeros((len(self.vocab), self.embed_size))
			self.get_emb_matrix_given_vocab()  # assign emb_matrix -> (len(vocab), embed_size)
			self.x_embed = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_id)
			self.x_embed = self.x_embed.from_pretrained(torch.FloatTensor(self.emb_matrix))
			self.x_embed.weight.data[self.pad_id] = 0.0 # zero padding

			# padding_indx is disappeared when we use "from_pretrained" in pytorch 1.0 (bug?)
			self.x_embed.padding_idx = self.pad_id

			self.embedding_dim = self.x_embed.embedding_dim

		if config.use_gpu and self.x_embed is not None:
			self.x_embed = utils.cast_type(self.x_embed, FLOAT, config.use_gpu)


		return
	# end __init__

	#
	def load_pretrained_bert(self):
		logger.info("Bert-Embedding")

		self.x_embed = BertModel.from_pretrained(self.emb_path).embeddings

		bert_tokenizer = BertTokenizer.from_pretrained(self.emb_path, do_lower_case=True)
		self.vocab = bert_tokenizer.vocab
		self.rev_vocab = bert_tokenizer.ids_to_tokens

		return self.x_embed
	# end load_pretrained_bert

	#
	def load_pretrained_file(self):
		has_header = False

		# read first line to identify whether it has header or not
		with codecs.open(self.emb_path, 'r', encoding='utf8') as emb_file:
			# tokens = emb_file.next().split()
			tokens = emb_file.readline().split()  # modified for python3
			if len(tokens) == 2:
				try:
					int(tokens[0])
					int(tokens[1])
					has_header = True
				except ValueError:
					pass

		#
		if has_header:
			with codecs.open(self.emb_path, 'r', encoding='utf8') as emb_file:
				# tokens = emb_file.next().split()
				tokens = emb_file.readline().split()  # modified for python3
				assert len(tokens) == 2, 'The first line in W2V embeddings must be the pair (vocab_size, embed_size)'
				self.vocab_size = int(tokens[0])
				self.embed_size = int(tokens[1])
				assert self.embed_size == self.embed_size, 'The embeddings dimension does not match with the requested dimension'
				counter = 0
				for line in emb_file:
					tokens = line.split()
					word = tokens[0]
					vec = np.array([float(f) for f in tokens[1].split(',')])
					assert len(vec) == self.embed_size, 'The number of dimensions does not match the header info'
					self.embeddings[word] = vec
					# self.vectors.append(vec)
					counter += 1
			assert counter == self.vocab_size, 'Vocab size does not match the header info'
			# self.dim = self.embed_size
			# self.vectors = np.array(self.vectors)
		#
		else:
			with codecs.open(self.emb_path, 'r', encoding='utf8') as emb_file:
				# self.vocab_size = 0
				self.embed_size = -1
				for line in emb_file:
					tokens = line.split()
					if self.embed_size == -1:
						self.embed_size = len(tokens) - 1
						assert self.embed_size == self.embed_size, 'The embeddings dimension does not match with the requested dimension'
					else:
						assert len(tokens) == self.embed_size + 1, 'The number of dimensions does not match the header info'
					word = tokens[0]
					vec = tokens[1:]
					self.embeddings[word] = vec
					self.vocab_size += 1
		logger.info('  #vectors: %i, #dimensions: %i' % (self.vocab_size, self.embed_size))

		return
	# end load_pretrained_file

	#
	def get_emb_given_word(self, word):
		try:
			return self.embeddings[word]
		except KeyError:
			return None
	# end get_emb_given_word

	# returns emb_matrix (numpy array); modified for pytorch
	def get_emb_matrix_given_vocab(self):
		counter = 0.

		# for word, _ in self.vocab.iteritems():
		for word in self.vocab:
			try:
				index = self.rev_vocab[word]
				if index > (self.num_special_vocab - 1): # skip the special index, e.g., PAD, EOS
					self.emb_matrix[index] = np.array(self.embeddings[word])
				else: # embedding for special index
					self.emb_matrix[index] = np.array([0.0] * self.embed_size) # zero init for special index

				counter += 1
			except KeyError:
				pass
				# print("KeyError!" + "\t" + word)

		logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(self.vocab), 100 * counter / len(self.vocab)))

		return self.emb_matrix
	# end get_emb_matrix_given_vocab

	#
	def get_embed_layer(self):
		return self.x_embed
	# end get_embed_size

# end class




