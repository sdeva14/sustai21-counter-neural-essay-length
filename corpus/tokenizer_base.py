import os

import nltk
# from pytorch_pretrained_bert.tokenization import BertTokenizer

# from pytorch_transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer

#
class Tokenizer_Base(object):

	def __init__(self, config):
		super().__init__()
		
		return
	# end __init__

		#
	def get_tokenizer(self, config):
		tokenizer = None
		# if not configured, then no need to assign
		if config.tokenizer_type.startswith('word'):
			tokenizer = nltk.word_tokenize
		elif config.tokenizer_type.startswith('bert-'):
			tokenizer = BertTokenizer.from_pretrained(config.tokenizer_type, do_lower_case=True)
		elif config.tokenizer_type.startswith('xlnet'):
			tokenizer = XLNetTokenizer.from_pretrained(config.tokenizer_type, do_lower_case=True)

		return tokenizer

