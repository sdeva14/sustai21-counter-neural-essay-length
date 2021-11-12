# -*- coding: utf-8 -*-

from __future__ import unicode_literals  # at top of module

import os
import logging
import re
import string
from collections import Counter
import statistics

import numpy as np
import pandas as pd
import math

from scipy.stats import entropy
from math import log, e

import nltk
# from pytorch_pretrained_bert.tokenization import BertTokenizer

from corpus.tokenizer_base import Tokenizer_Base
import sentencepiece as spm

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
SEP = "|"
TIME= '<time>'
DATE = '<date>'

# import spacy
# spacy_nlp = spacy.load('en')
# import nltk.tokenize.punkt

# custom sentence splitter for testing
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"

time_regex1 = re.compile(u'^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$')
time_regex2 = re.compile(u'^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]?([AaPp][Mm])$')

date_regex1 = re.compile(u'(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.][0-9]{2}')
date_regex2 = re.compile(u'(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.](19|20)[0-9]{2}')
date_regex3 = re.compile(u'((19|20)[- /.][0-9]{2}0[1-9]|1?[012])[- /.](0[1-9]|[12][0-9]|3[01])')
date_regex4 = re.compile(u'((19|20)[- /.][0-9]0[1-9]|1?[0-9])[- /.](0[1-9]|[12][0-9]|3[01])')

def split_into_sentences(text):
    """ customized sentence splitter for testing (not used now) """

    text = " " + text + "  "
    text = text.replace("\n\n", " . ")  # ignore paragraph here
    text = text.replace("\n"," . ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "\"" in text: text = text.replace(".\"","\".")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")

    text = text.replace(" and i ", " and i<stop>")
    text = text.replace(" and she ", " and she<stop>")
    text = text.replace(" and he ", " and he<stop>")
    text = text.replace(" and we ", " and we<stop>")
    text = text.replace(" and they ", " and they<stop>")
    text = text.replace(" and their ", " and their<stop>")
    text = text.replace(" and my ", " and my<stop>")
    text = text.replace(" and her ", " and her<stop>")
    text = text.replace(" and his ", " and his<stop>")
    text = text.replace(", and", ", and<stop>")
    text = text.replace(": ", ":<stop>")

    text = text.replace(" of course ", " of course<stop>")

    text = text.replace("<prd>",".")

    text = re.sub("\((.?)\)", "<stop>", text)  # (a), (b), (1), (2)
    text = re.sub("( .?)\)", "<stop>", text)  # a), b), 1), 2)

    text = text.replace("e.g. ,", "e.g. ,<stop>")
    text = text.replace("e.g ,", "e.g. ,<stop>")

    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace(";", ";<stop>")
    text = text.replace("*", "*<stop>")
    text = text.replace(" - ", "-<stop>")

    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]

    return sentences
# end def split_into_sentences


class CorpusBase(object):
    """ Corpus class for base """

    #
    def __init__(self, config):
        super(CorpusBase, self).__init__()
        self.config = config

        # self.token_type = config.token_type # word or sentpiece
        self.tokenizer_type = config.tokenizer_type # nltk or bert-base-uncased

        self.vocab = None  # will be assigned in "_build_vocab" i.e., word2ind
        self.rev_vocab = None  # will be assigned in "_build_vocab" i.e., ind2word
        self.pad_id = 0  # default 0, will be re-assigned depending on tokenizer
        self.unk_id = None  # will be assigned in "_build_vocab"
        self.bos_id = None  # will be assigned in "_build_vocab"
        self.eos_id = None  # will be assigned in "_build_vocab"
        self.time_id = None
        self.vocab_count = -1  # will be assigned in "_build_vocab"
        self.num_special_vocab = None  # number of used additional vocabulary, e.g., PAD, UNK, BOS, EOS

        self.train_corpus = None  # will be assigned in "read_kfold"
        self.valid_corpus = None  # will be assigned in "read_kfold"
        self.test_corpus = None  # will be assigned in "read_kfold"

        self.fold_train = []  # (num_fold, structured_train), # will be assigned in "read_kfold"
        self.fold_test = []  # (num_fold, structured_test), # will be assigned in "read_kfold"
        # cols= ['ind_origin', 'text_id', 'subject', 'text', 'ratingA1', 'ratingA2', 'ratingA3', 'labelA', 'ratingM1', 'ratingM2', 'ratingM3', 'ratingM4', 'ratingM5', 'labelM']
        self.cur_fold_num = -1  #

        self.max_num_sents = -1  # maximum number of sentence in document given corpus, will be assigned in "_read_dataset"
        self.max_len_sent = -1  # maximum length of sentence given corpus, will be assigned in "_read_dataset"
        self.max_len_doc = -1  # maximum length of documents (the number of words), will be assigned in "_read_dataset"

        self.output_bias = None

        self.keep_pronoun = config.keep_pronoun
        self.remove_stopwords = config.remove_stopwords
        self.stopwords = []

        # get tokenizer
        # self.tokenizer = self._get_tokenizer(config)
        tokenizer_class = Tokenizer_Base(config)
        self.tokenizer = tokenizer_class.get_tokenizer(config)

        # sentence splitter
        self.sent_tokenzier = nltk.sent_tokenize  # nltk sent tokenizer

        # stopwords
        self._make_stopwords()

    ##########################

    # #
    # def _get_tokenizer(self, config):
    #     tokenizer = None
    #     # if not configured, then no need to assign
    #     if self.tokenizer_type.startswith('bert-'):
    #         tokenizer = BertTokenizer.from_pretrained(self.tokenizer_type, do_lower_case=True)
    #     elif self.tokenizer_type.startswith('word'):
    #         tokenizer = nltk.word_tokenize

    #     return tokenizer

    #
    def set_cur_fold_num(self, cur_fold_num):
        self.cur_fold_num = cur_fold_num
        return

    #
    def get_id_corpus(self, num_fold=-1):
        raise NotImplementedError

    #
    def _tokenize_corpus(self, pd_input):
        raise NotImplementedError

    #
    def _read_dataset(self, config):
        raise NotImplementedError

    #
    def generate_kfold(self, config, seed):
        raise NotImplementedError

    #
    def read_kfold(self, config):
        raise NotImplementedError

    #
    def is_time(self, token):
        is_time = False
        if bool(time_regex1.match(token)): is_time = True
        elif bool(time_regex2.match(token)): is_time = True

        return is_time

    #
    def is_date(self, token):
        is_date = False
        if bool(date_regex1.match(token)): is_date = True
        elif bool(date_regex2.match(token)): is_date = True
        elif bool(date_regex3.match(token)): is_date = True
        elif bool(date_regex4.match(token)): is_date = True

        return is_date

    #
    def _build_vocab(self, max_vocab_cnt):
        # build vocab
        if self.tokenizer_type.startswith('word'):
            self._build_vocab_manual(max_vocab_cnt)
        elif self.tokenizer_type.startswith('bert-'):
            self.vocab = self.tokenizer.vocab
            self.rev_vocab = self.tokenizer.ids_to_tokens
            self.pad_id = self.vocab["[PAD]"]
            self.vocab_count = 30522  # fixed for pretrained BERT vocab
        elif self.tokenizer_type.startswith('xlnet-'):
            # self.vocab = self.tokenizer.vocab
            # self.rev_vocab = self.tokenizer.ids_to_tokens
            # self.pad_id = self.vocab["[PAD]"]
            self.pad_id = 0
            self.vocab_count = 32000  # fixed for pretrained BERT vocab

            s = spm.SentencePieceProcessor()
            spiece_model = "xlnet-base-cased-spiece.model"
            s.Load(spiece_model)

            map_vocab = {}
            for ind in range(32000):
                map_vocab[ind] = s.id_to_piece(ind)

            inv_map = {v: k for k, v in map_vocab.items()}

            self.vocab = map_vocab
            self.rev_vocab = inv_map


        return

    #
    def _build_vocab_manual(self, max_vocab_cnt):
        """tokenize to word level for building vocabulary"""

        all_words = []
        for cur_doc in self.train_corpus:
            for cur_sent in cur_doc:
                tokenized_words = nltk.word_tokenize(cur_sent)
                all_words.extend(tokenized_words)

        vocab_count = Counter(all_words).most_common()
        vocab_count = vocab_count[0:max_vocab_cnt]

        # # create vocabulary list sorted by count for printing
        # raw_vocab_size = len(vocab_count)  # for printing
        # discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])  # for printing
        # print("Load corpus with train size %d, valid size %d, "
        #       "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
        #       % (len(self.train_corpus), len(self.valid_corpus),
        #          len(self.test_corpus),
        #          raw_vocab_size, len(vocab_count), vocab_count[-1][1],
        #          float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, BOS, EOS, TIME, DATE] + [t for t, cnt in
                                                         vocab_count]  # insert BOS and EOS to sentence later actually
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}

        self.pad_id = self.rev_vocab[PAD]
        self.unk_id = self.rev_vocab[UNK]
        self.bos_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.time_id = self.rev_vocab[TIME]
        self.date_id = self.rev_vocab[DATE]

        self.num_special_vocab = len(self.vocab) - max_vocab_cnt
        self.vocab_count = len(self.vocab)

        return
    # end def _build_vocab

    #
    def _get_stat_corpus(self):
        """ get statistics required for seq2seq processing from stored corpus"""
        
        ## get the number of sents in given whole corpus, regardless of train or test
        list_num_sent_doc = [len(doc) for doc in self.train_corpus]
        list_num_sent_doc = list_num_sent_doc + [len(doc) for doc in self.test_corpus]
        if self.valid_corpus is not None:
            list_num_sent_doc = list_num_sent_doc + [len(doc) for doc in self.valid_corpus]

        self.avg_num_sents = statistics.mean(list_num_sent_doc)
        self.std_num_sents = statistics.stdev(list_num_sent_doc)
        self.max_num_sents = np.max(list_num_sent_doc)  # document length (in terms of sentences)

        # print("Num Sents")
        # print(str(self.max_num_sents) + "\t" + str(self.avg_num_sents) + "\t" + str(self.std_num_sents))
        # print()

        ## get length of sentences
        self.max_len_sent = 0
        if self.tokenizer_type.startswith("bert") or self.tokenizer_type.startswith("xlnet"):
            list_len_sent = [len(self.tokenizer.tokenize(sent)) for cur_doc in self.train_corpus for sent in cur_doc]
            list_len_sent = list_len_sent + [len(self.tokenizer.tokenize(sent)) for cur_doc in self.test_corpus for sent in cur_doc]
            if self.valid_corpus is not None:
                list_len_sent = list_len_sent + [len(self.tokenizer.tokenize(sent)) for cur_doc in self.valid_corpus for sent in cur_doc]

        else:
            list_len_sent = [len(nltk.word_tokenize(sent)) for cur_doc in self.train_corpus for sent in cur_doc]
            list_len_sent = list_len_sent + [len(nltk.word_tokenize(sent)) for cur_doc in self.test_corpus for sent in cur_doc]
            if self.valid_corpus is not None:
                list_len_sent = list_len_sent + [len(nltk.word_tokenize(sent)) for cur_doc in self.valid_corpus for sent in cur_doc]

        self.max_len_sent = np.max(list_len_sent)
        self.max_len_sent = self.max_len_sent + 2  # because of special character BOS and EOS (or SEP)
        self.avg_len_sent = statistics.mean(list_len_sent)
        self.std_len_sent = statistics.stdev(list_len_sent)

        # print("Len Sent")
        # print(str(self.max_len_sent-2) + "\t" + str(self.avg_len_sent) + "\t" + str(self.std_len_sent))
        # print()

        ## get document length (in terms of words number)
        list_len_word_doc = self._get_list_len_word_doc(self.train_corpus)
        list_len_word_doc = list_len_word_doc + self._get_list_len_word_doc(self.test_corpus)
        if self.valid_corpus is not None:
            list_len_word_doc = list_len_word_doc + self._get_list_len_word_doc(self.valid_corpus)

        self.max_len_doc = np.max(list_len_word_doc)
        self.avg_len_doc = statistics.mean(list_len_word_doc)
        self.std_len_doc = statistics.stdev(list_len_word_doc)

        # print("Len Doc")
        # print(str(self.max_len_doc) + "\t" + str(self.avg_len_doc) + "\t" + str(self.std_len_doc))
        # print()


        return

    #
    def _get_max_doc_len(self, target_corpus):
        """ get maximum document length for seq2seq """

        doc_len_list = []
        for cur_doc in target_corpus:
            if self.tokenizer_type.startswith("bert") or self.tokenizer_type.startswith("xlnet"):
                len_num_words = len(self.tokenizer.tokenize(' '.join(sent for sent in cur_doc)))
                doc_len_list.append(len_num_words + (len(cur_doc)))
            else:
                cur_text = ' '.join(sent for sent in cur_doc)
                len_num_words = len(nltk.word_tokenize(cur_text))
                doc_len_list.append(len_num_words + (len(cur_doc)*2) )  # should be considered that each sent has bos and eos

        return max(doc_len_list)

    #
    def _get_list_len_word_doc(self, target_corpus):
        """ get maximum document length for seq2seq """

        doc_len_list = []
        for cur_doc in target_corpus:
            if self.tokenizer_type.startswith("bert") or self.tokenizer_type.startswith("xlnet"):
                len_num_words = len(self.tokenizer.tokenize(' '.join(sent for sent in cur_doc)))
                doc_len_list.append(len_num_words + (len(cur_doc)))
            else:
                cur_text = ' '.join(sent for sent in cur_doc)
                len_num_words = len(nltk.word_tokenize(cur_text))
                doc_len_list.append(len_num_words + (len(cur_doc)*2) )  # should be considered that each sent has bos and eos

        return doc_len_list

    #
    def _refine_text(self, input_text, ignore_uni=True, ignore_para=True):
        """ customized function for pre-processing raw text"""
        input_text = input_text.lower()
        out_text = input_text

        return out_text

    #
    def _make_stopwords(self):
        """ make stopwords list (not used now)"""

        # snowball stopwords
        snowball_stopwords = "i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing would should could ought i'm you're he's she's it's we're they're i've you've we've they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't doesn't don't didn't won't wouldn't shan't shouldn't can't cannot couldn't mustn't let's that's who's what's here's there's when's where's why's how's a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very"
        stopwords = snowball_stopwords.split()

        if not self.keep_pronoun:
            pronouns = ['i', 'me', 'we', 'us', 'you', 'she', 'her', 'him', 'he', 'it', 'they', 'them', 'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves']
            stopwords = list(set(stopwords) - set(pronouns))

        str_punct = [t for t in string.punctuation]
        stopwords = stopwords + str_punct
        stopwords = stopwords + [u'``',u"''",u"lt",u"gt", u"<NUM>"]

        stopwords.remove('.')

        self.stopwords = stopwords

        return
    # end _make_stopwords

    #
    def _sent_split_corpus(self, arr_input_text):
        """ tokenize corpus given tokenizer by config file"""
        # arr_input_text = pd_input['essay'].values

        # num_over = 0
        # total_sent = 0

        sent_corpus = []  # tokenized to form of [doc, list of sentences]
        for cur_doc in arr_input_text:
            cur_doc = self._refine_text(cur_doc)  # cur_doc: single string
            sent_list = self.sent_tokenzier(cur_doc)  # following exactly same way with previous works
            # sent_list = [sent.string.strip() for sent in spacy_nlp(cur_doc).sents] # spacy style
            # sent_list = corpus.corpus_base.split_into_sentences(refined_doc) # customized style (nltk and spacy does not work well on GCDC)
            sent_list = [x for x in sent_list if len(nltk.word_tokenize(x)) > 1]  # if there is mis-splitted sentence, then remove e.g., "."
            sent_corpus.append(sent_list)

            # # BEA19 test (they only consider less than 25 sentences)
            # if len(sent_list) > 25:
            #     sent_list = sent_list[:25]
            #     # num_over = num_over + 1

        #     if len(sent_list) > 25:
        #         num_over = num_over + len(sent_list) - 25

        #     sent_list = sent_list[:25]
        #     for cur_sent in sent_list:
        #         words = nltk.word_tokenize(cur_sent)
        #         if len(words) > 40:
        #             num_over = num_over + 1
        #     total_sent = total_sent + len(sent_list)

        # print("Over Sent: " + str(num_over))                
        # print("Total Sent: " + str(total_sent))            



        return sent_corpus

    #
    def get_avg_entropy(self):
        min_rating, max_rating = self.score_ranges[self.prompt_id_train]

        ##
        # scores = []
        all_entropies = []
        for cur_score in range(min_rating, max_rating+1):
            # cur_score_pd = self.merged_pd.loc[self.merged_pd['essay_score'] == cur_score]
            cur_score_pd = self.train_pd.loc[self.train_pd['essay_score'] == cur_score]

            if len(cur_score_pd) < 1: continue
            # print(len(cur_score_pd))

            essays = cur_score_pd['essay'].values

            essays = self._sent_split_corpus(essays)
            id_essays, _, _= self._to_id_corpus(essays)

            # entropies = []
            # get entropy
            for cur_id_essay in id_essays:
                cur_id_essay = [x for sublist in cur_id_essay for x in sublist] # flatten

                value,counts = np.unique(cur_id_essay, return_counts=True)
                # norm_counts = counts / counts.sum()
                norm_counts = counts / float(4000)
                base = e
                ent = -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

                # entropies.append(ent)
                all_entropies.append(ent)
                # scores.append(cur_score)

            # end for cur_id_essay

        # end for cur_score
        avg_ent = math.ceil(statistics.mean(all_entropies))

        return avg_ent

    #
    def get_p_kld(self):

        ##
        min_rating, max_rating = self.score_ranges[self.prompt_id_train]
        all_ids = []
        # ratio_high_score = 0.67

        high_rating = math.ceil(max_rating * self.ratio_high_score)
        for cur_score in range(high_rating, max_rating+1):
            cur_score_pd = self.train_pd.loc[self.train_pd['essay_score'] == cur_score]

            if len(cur_score_pd) < 1: continue
            # print(len(cur_score_pd))

            essays = cur_score_pd['essay'].values

            essays = self._sent_split_corpus(essays)
            id_essays, _, _ = self._to_id_corpus(essays)

            # get entropy
            entropies = []
            for cur_id_essay in id_essays:
                cur_id_essay = [x for sublist in cur_id_essay for x in sublist] # flatten

                all_ids.extend(cur_id_essay)

        # end for cur_score

        # get probability distribution of P for KL-D from high scored essays
        all_ids_counter = Counter(all_ids)
        total_ids_num = len(all_ids)
        p_map = dict.fromkeys(self.rev_vocab, 0.0000000001)  # if it is not xlnet
        list_vocab = list(self.rev_vocab)
        for cur_id, cur_cnt in all_ids_counter.items():
            p_map[list_vocab[cur_id]] = cur_cnt / total_ids_num
        # print(p_map)

        # iterate all essays again to get KL-d with Q distribution
        all_kld = []
        # for cur_score in range(min_rating, max_rating+1):
        for cur_score in range(high_rating, max_rating+1):
            list_kld_score = []
            cur_score_pd = self.train_pd.loc[self.train_pd['essay_score'] == cur_score]
            if len(cur_score_pd) < 1: continue

            essays = cur_score_pd['essay'].values
            essays = self._sent_split_corpus(essays)
            id_essays, _, _ = self._to_id_corpus(essays)

            for cur_id_essay in id_essays:
                cur_id_essay = [x for sublist in cur_id_essay for x in sublist] # flatten
                cur_ids_counter = Counter(cur_id_essay)
                q_map = dict.fromkeys(self.rev_vocab, 0.0000000001)
                list_vocab = list(self.rev_vocab)

                # get Q distribution from current input essay
                for cur_id, cur_cnt in cur_ids_counter.items():
                    q_map[list_vocab[cur_id]] = cur_cnt / len(cur_id_essay)

                # get KL_d for each essay
                # print(q_map.values())
                cur_kld = entropy(pk=list(p_map.values()), qk=list(q_map.values()))
                # cur_kld = (np.array(list(p_map.values())) * np.log(np.array(list(p_map.values()))/np.array(list(q_map.values())))).sum()
                list_kld_score.append(cur_kld)

                # scores.append(cur_score)
                all_kld.append(cur_kld)
            # end for cur_id_essay
        # end for cur_score

        total_avg_kld = statistics.mean(all_kld)

        return p_map, total_avg_kld

    # get KL-divergence as given essay score range
    def get_kld_range(self, min_rating, high_rating, total_vocab):
        ## get p map first in training set
        # ratio_high_score = 0.8  ## defined in each target class
        # high_rating = math.ceil((max_rating - min_rating) * self.ratio_high_score + min_rating)
        all_tokens = []
        for cur_score in range(min_rating, high_rating+1):
            cur_score_pd = self.train_pd.loc[self.train_pd['essay_score'] == cur_score]

            if len(cur_score_pd) < 1: continue

            essays = cur_score_pd['essay'].values

            # for cur_id_essay in id_essays:
            for cur_essay in essays:
                # cur_id_essay = [x for sublist in cur_id_essay for x in sublist] # flatten
                cur_essay = cur_essay.lower()
                tokens_essay = nltk.word_tokenize(cur_essay)
                all_tokens.extend(tokens_essay)
            # end for cur_id_essay
        # end for cur_score

        ## get probability distribution of P for KL-D from high scored essays
        all_tokens_counter = Counter(all_tokens)
        total_tokens_num = len(all_tokens)

        p_map = dict.fromkeys(total_vocab, 0.0000000001)
        for cur_id, cur_cnt in all_tokens_counter.items():
            p_map[cur_id] = float(cur_cnt) / total_tokens_num

        ## get average KL-divergence in training set
        all_kld = []
        # for cur_score in range(min_rating, max_rating+1):
        for cur_score in range(min_rating, high_rating+1):
            list_kld_score = []
            cur_score_pd = self.train_pd.loc[self.train_pd['essay_score'] == cur_score]
            if len(cur_score_pd) < 1: continue
            essays = cur_score_pd['essay'].values
            
            for cur_essay in essays:
                cur_essay = cur_essay.lower()
                tokens_essay = nltk.word_tokenize(cur_essay)
                cur_tokens_counter = Counter(tokens_essay)

                q_map = dict.fromkeys(total_vocab, 0.0000000001)

                # get Q distribution from current input essay
                for cur_id, cur_cnt in cur_tokens_counter.items():
                    q_map[cur_id] = cur_cnt / len(tokens_essay)

                # get KL_d for each essay
                cur_kld = entropy(pk=list(p_map.values()), qk=list(q_map.values()))
                all_kld.append(cur_kld)
            # end for cur_essay
        # end for cur_score
        total_avg_kld = statistics.mean(all_kld)

        # get KL-divergence for each essay
        map_kld_essays = dict()  # key: essay_id, value: kld
        all_essay_data = self.merged_pd.loc[:, ['essay', 'essay_id']].values
        for cur_essay, cur_essay_id in all_essay_data:
            cur_essay = cur_essay.lower()
            tokens_essay = nltk.word_tokenize(cur_essay)
            cur_tokens_counter = Counter(tokens_essay)

            # get Q distribution from current input essay
            q_map = dict.fromkeys(total_vocab, 0.0000000001)
            for cur_id, cur_cnt in cur_tokens_counter.items():
                q_map[cur_id] = cur_cnt / len(tokens_essay)

            # get kld and store to the map
            cur_kld = entropy(pk=list(p_map.values()), qk=list(q_map.values()))
            map_kld_essays[cur_essay_id] = cur_kld / total_avg_kld
            # map_kld_essays[cur_essay_id] = cur_kld

        return map_kld_essays

    # get KL-divergence in word level in advance
    def get_word_kld(self):
        min_rating, max_rating = self.score_ranges[self.prompt_id_train]

        ## make a word dictionary for whole essays
        # total_essays = self.merged_pd['essay'].values
        total_essays = self.merged_pd['essay'].values
        total_tokens = []
        for cur_essay in total_essays:
            cur_essay = cur_essay.lower()
            tokens_essay = nltk.word_tokenize(cur_essay)
            total_tokens.append(tokens_essay)

        total_tokens = [x for sublist in total_tokens for x in sublist] # flatten
        total_tokens_counter = Counter(total_tokens)
        total_vocab = dict()
        for cur_id, cur_cnt in total_tokens_counter.items():
            total_vocab[cur_id] = cur_cnt / float(len(total_tokens))

        # get kld for each essay
        mid_rating = math.floor(((max_rating - min_rating) * self.ratio_mid_score) + min_rating)
        high_rating = math.ceil((max_rating - min_rating) * self.ratio_high_score + min_rating)
        map_kld_essays_low = self.get_kld_range(min_rating, mid_rating, total_vocab)
        map_kld_essays_mid = self.get_kld_range(mid_rating, high_rating, total_vocab)
        map_kld_essays_high = self.get_kld_range(high_rating, max_rating, total_vocab)
        map_kld_essays = {key: [value] + [map_kld_essays_mid[key]] + [map_kld_essays_high[key]] for key, value in map_kld_essays_low.items()}


        return map_kld_essays


    #
    def _to_id_corpus(self, data):
        """
        Get id-converted corpus
        :param data: corpus data
        :return: id-converted corpus
        """
        results = []
        max_len_doc = -1
        list_doc_len = []
        entropies = []
        kld = []
        for cur_doc in data:
            temp = []
            for raw_sent in cur_doc:
                id_sent = self._sent2id(raw_sent)  # convert to id

                temp.append(id_sent)
            results.append(temp)

            # save max doc len
            flat_doc = [item for sublist in temp for item in sublist]
            # if len(flat_doc) > max_len_doc:
            #     max_len_doc = len(flat_doc)
            list_doc_len.append(len(flat_doc))

        #
        max_len_doc = np.max(list_doc_len)
        avg_len_doc = math.ceil(statistics.mean(list_doc_len))

        # return results, max_len_doc, avg_len_doc
        return results, max_len_doc, avg_len_doc

    #
    def _sent2id(self, sent):
        """ return id-converted sentence """

        # note that, it is not zero padded yet here
        id_sent = []
        if self.tokenizer_type.startswith("word"):
            tokens_sent = nltk.word_tokenize(sent)  # word level tokenizer

            id_sent = [self.rev_vocab.get(t, self.unk_id) for t in tokens_sent]
            id_sent = [self.bos_id] + id_sent + [self.eos_id]  # add BOS and EOS to make an id-converted sentence
        elif self.tokenizer_type.startswith("bert"):
            # tokens_sent = self.tokenizer.tokenize(sent)
            # #tokens_sent = ["[CLS]"] + tokens_sent + ["[SEP]"]  # CLS is required when we use classification task in BERT
            # tokens_sent = tokens_sent + ["[SEP]"]  # otherwise, SEP is enough
            # id_sent = self.tokenizer.convert_tokens_to_ids(tokens_sent)
            id_sent = self.tokenizer.encode(sent)
        elif self.tokenizer_type.startswith("xlnet"):
            # id_sent = torch.tensor([tokenizer.encode(sent)])
            id_sent = self.tokenizer.encode(sent)        

        return id_sent
