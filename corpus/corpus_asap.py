# -*- coding: utf-8 -*-

import logging
import os, io

from collections import Counter
import pandas as pd
import numpy as np
import sklearn.model_selection
import nltk
from unidecode import unidecode
import statistics

import corpus.corpus_base
from corpus.corpus_base import CorpusBase
from corpus.corpus_base import PAD, UNK, BOS, EOS, BOD, EOD, SEP, TIME, DATE

logger = logging.getLogger()

from corpus.dataset_asap import Dataset_ASAP

class CorpusASAP(CorpusBase):

    #
    def __init__(self, config):
        super(CorpusASAP, self).__init__(config)

        self.score_ranges = {
            1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60)
        }
        if config.output_size < 0:
            config.output_size = 1  # MSE loss setting

        self.train_total_pd = None
        self.valid_total_pd = None
        self.test_total_pd = None

        self.train_pd = None
        self.valid_pd = None
        self.test_pd = None

        # self.cur_prompt_id = config.essay_prompt_id
        self.prompt_id_train = config.essay_prompt_id_train
        self.prompt_id_test = config.essay_prompt_id_test

        # self.ratio_high_score = 0.8
        # self.ratio_high_score = 0.67
        self.ratio_high_score = 0.8
        self.ratio_mid_score = 0.3

        if self.prompt_id_train == 7:  ## due to lack of essays in low scores
            self.ratio_mid_score = 0.35
        if self.prompt_id_train == 8:  ## due to lack of essays in low scores
            self.ratio_mid_score = 0.4

        self.k_fold = config.num_fold

        # self.train_corpus = None  # just for remind, already declared in the corpus_base
        # self.test_corpus = None

        if config.is_gen_cv:
            seed = np.random.seed()
            self.generate_kfold(config=config, seed=seed)  # generate k-fold file

        self._read_dataset(config)  # store whole dataset as pd.dataframe

        self._build_vocab(config.max_vocab_cnt)

        return 
    # end __init__

    #
    def _get_model_friendly_scores(self, essay_pd, prompt_id_target):
        """ scale between 0 to 1 for MSE loss function"""

        # scores_array = essay_pd['domain1_score'].values
        scores_array = essay_pd['essay_score'].values
        min_rating, max_rating = self.score_ranges[prompt_id_target]
        scaled_label = (scores_array - min_rating) / (max_rating - min_rating)

        essay_pd.insert(len(essay_pd.columns), 'rescaled_label', scaled_label)

        return essay_pd

    #
    def _relabel_p(self, essay_pd, prompt_id_target):
        """testing function for relabeling prompt 8"""

        cur_relabel_ranges = self.relabel_ranges[prompt_id_target]
        min_rating, max_rating = self.score_ranges[prompt_id_target]

        scores_array = essay_pd['domain1_score'].values
        for ind, cur_val in enumerate(scores_array):
            # check relabel group
            cur_relabel = -1
            for relabel_key in cur_relabel_ranges:
                cur_min, cur_max = cur_relabel_ranges[relabel_key]
                if cur_val >= cur_min and cur_val <= cur_max:
                    cur_relabel = relabel_key
                    break
            # end for
            
            # update for relabeling
            if cur_relabel > -1:
                scores_array[ind] = cur_relabel
        # end for

        # update relabeld array to dataframe
        essay_pd = essay_pd.assign(domain1_score=scores_array)

        return essay_pd
    # end def _relabel_p

    #
    def _read_dataset(self, config):
        """ read asap dataset, assumed that splitted to "train.tsv", "dev.tsv", "test.tsv" under "fold_" """

        path_asap = config.data_dir
        cur_path_asap = os.path.join(path_asap, "fold_" + str(config.cur_fold))

        # # read
        self.train_total_pd = pd.read_csv(os.path.join(cur_path_asap, "train.tsv"), sep="\t", header=0, encoding="utf-8", engine='c')  # skip the first line
        self.valid_total_pd = pd.read_csv(os.path.join(cur_path_asap, "dev.tsv"), sep="\t", header=0, encoding="utf-8", engine='c')
        self.test_total_pd = pd.read_csv(os.path.join(cur_path_asap, "test.tsv"), sep="\t", header=0, encoding="utf-8", engine='c')

        # # spell-checked test
        # self.train_total_pd = pd.read_csv(os.path.join(cur_path_asap, "train_spelled.tsv"), sep="\t", header=0, encoding="utf-8", engine='c')  # skip the first line
        # self.valid_total_pd = pd.read_csv(os.path.join(cur_path_asap, "dev_spelled.tsv"), sep="\t", header=0, encoding="utf-8", engine='c')
        # self.test_total_pd = pd.read_csv(os.path.join(cur_path_asap, "test_spelled.tsv"), sep="\t", header=0, encoding="utf-8", engine='c')

        train_pd = self.train_total_pd.loc[self.train_total_pd['essay_set'] == self.prompt_id_train]
        valid_pd = self.valid_total_pd.loc[self.valid_total_pd['essay_set'] == self.prompt_id_test]
        test_pd = self.test_total_pd.loc[self.test_total_pd['essay_set'] == self.prompt_id_test]

        # for fun: relabel
        #self.train_pd = self._relabel_p(train_pd, self.prompt_id_train)
        #self.valid_pd = self._relabel_p(valid_pd, self.prompt_id_test)
        #self.test_pd = self._relabel_p(test_pd, self.prompt_id_test)

        train_pd = train_pd.rename({'domain1_score': 'essay_score'}, axis='columns')
        valid_pd = valid_pd.rename({'domain1_score': 'essay_score'}, axis='columns')
        test_pd = test_pd.rename({'domain1_score': 'essay_score'}, axis='columns')

        # scaling score for training due to different score range as prompt_id
        self.train_pd = self._get_model_friendly_scores(train_pd, self.prompt_id_train)
        self.valid_pd = self._get_model_friendly_scores(valid_pd, self.prompt_id_test)
        self.test_pd = self._get_model_friendly_scores(test_pd, self.prompt_id_test)

        self.merged_pd = pd.concat([self.train_pd, self.valid_pd, self.test_pd])
        # self.merged_pd = self.merged_pd.rename({'domain1_score': 'essay_score', 'essay_set': 'prompt'}, axis='columns')

        # put bias as mean of labels
        self.output_bias = self.train_pd['rescaled_label'].values.mean(axis=0)

        # sentence split
        self.train_corpus = self._sent_split_corpus(train_pd['essay'].values)  # (doc_id, list of sents)
        self.valid_corpus = self._sent_split_corpus(valid_pd['essay'].values)
        self.test_corpus = self._sent_split_corpus(test_pd['essay'].values)

        # get statistics for training later
        self._get_stat_corpus()

        return
    # end _read_dataset

    # def _get_stat_corpus(self):
    #     """ get statistics required for seq2seq processing"""

    #     ## get the number of sents in given whole corpus, regardless of train or test
    #     list_num_sent_doc = [len(doc) for doc in self.train_corpus]
    #     list_num_sent_doc = list_num_sent_doc + [len(doc) for doc in self.test_corpus]
    #     if self.valid_corpus is not None:
    #         list_num_sent_doc = list_num_sent_doc + [len(doc) for doc in self.valid_corpus]

    #     self.avg_num_sents = statistics.mean(list_num_sent_doc)
    #     self.std_num_sents = statistics.stdev(list_num_sent_doc)
    #     self.max_num_sents = np.max(list_num_sent_doc)  # document length (in terms of sentences)

    #     print("Num Sents")
    #     print(str(self.max_num_sents) + "\t" + str(self.avg_num_sents) + "\t" + str(self.std_num_sents))
    #     print()

    #     ## get length of sentences
    #     self.max_len_sent = 0
    #     if self.tokenizer_type.startswith("bert") or self.tokenizer_type.startswith("xlnet"):
    #         list_len_sent = [len(self.tokenizer.tokenize(sent)) for cur_doc in self.train_corpus for sent in cur_doc]
    #         list_len_sent = list_len_sent + [len(self.tokenizer.tokenize(sent)) for cur_doc in self.test_corpus for sent in cur_doc]
    #         if self.valid_corpus is not None:
    #             list_len_sent = list_len_sent + [len(self.tokenizer.tokenize(sent)) for cur_doc in self.valid_corpus for sent in cur_doc]

    #     else:
    #         list_len_sent = [len(nltk.word_tokenize(sent)) for cur_doc in self.train_corpus for sent in cur_doc]
    #         list_len_sent = list_len_sent + [len(nltk.word_tokenize(sent)) for cur_doc in self.test_corpus for sent in cur_doc]
    #         if self.valid_corpus is not None:
    #             list_len_sent = list_len_sent + [len(nltk.word_tokenize(sent)) for cur_doc in self.valid_corpus for sent in cur_doc]

    #     self.max_len_sent = np.max(list_len_sent)
    #     self.max_len_sent = self.max_len_sent + 2  # because of special character BOS and EOS (or SEP)
    #     self.avg_len_sent = statistics.mean(list_len_sent)
    #     self.std_len_sent = statistics.stdev(list_len_sent)

    #     print("Len Sent")
    #     print(str(self.max_len_sent-2) + "\t" + str(self.avg_len_sent) + "\t" + str(self.std_len_sent))
    #     print()

    #     ## get document length (in terms of words number)
    #     list_len_word_doc = self._get_list_len_word_doc(self.train_corpus)
    #     list_len_word_doc = list_len_word_doc + self._get_list_len_word_doc(self.test_corpus)
    #     if self.valid_corpus is not None:
    #         list_len_word_doc = list_len_word_doc + self._get_list_len_word_doc(self.valid_corpus)

    #     self.max_len_doc = np.max(list_len_word_doc)
    #     self.avg_len_doc = statistics.mean(list_len_word_doc)
    #     self.std_len_doc = statistics.stdev(list_len_word_doc)

    #     print("Len Doc")
    #     print(str(self.max_len_doc) + "\t" + str(self.avg_len_doc) + "\t" + str(self.std_len_doc))
    #     print()

    #     return
    # # end def _get_stat_corpus

    #
    def get_id_corpus(self, num_fold=-1):
        """
        return id-converted corpus which is read in the earlier stage
        :param num_fold:
        :return: map of id-converted sentence
        """

        train_corpus = None
        valid_corpus = None
        test_corpus = None
        y_train = None
        y_valid = None
        y_test = None

        # ASAP is already divided
        train_corpus = self.train_corpus
        valid_corpus = self.valid_corpus
        test_corpus = self.test_corpus

        # change each sentence to id-parsed
        x_id_train, max_len_doc_train, avg_len_doc_train = self._to_id_corpus(train_corpus)
        x_id_valid, max_len_doc_valid, avg_len_doc_valid = self._to_id_corpus(valid_corpus)
        x_id_test, max_len_doc_test, avg_len_doc_test = self._to_id_corpus(test_corpus)

        max_len_doc = max(max_len_doc_train, max_len_doc_valid, max_len_doc_test)
        avg_len_doc = avg_len_doc_train

        # # debugging: max doc len in test
        # len_docs = []
        # for cur_doc in x_id_test:
        #     flat_doc = [item for sublist in cur_doc for item in sublist]
        #     len_docs.append(len(flat_doc))
        # max_len = max(len_docs)

        y_train = self.train_pd['rescaled_label'].values
        y_valid = self.valid_pd['rescaled_label'].values
        y_test = self.test_pd['rescaled_label'].values

        # score_train = self.train_pd['domain1_score'].values
        # score_valid = self.valid_pd['domain1_score'].values
        # score_test = self.test_pd['domain1_score'].values

        score_train = self.train_pd['essay_score'].values
        score_valid = self.valid_pd['essay_score'].values
        score_test = self.test_pd['essay_score'].values

        tid_train = self.train_pd['essay_id'].values
        tid_valid = self.valid_pd['essay_id'].values
        tid_test = self.test_pd['essay_id'].values

        train_data_id = {'x_data': x_id_train, 'y_label': y_train, 'tid': tid_train, 'origin_score': score_train}  # origin score is needed for qwk
        valid_data_id = {'x_data': x_id_valid, 'y_label': y_valid, 'tid': tid_valid, 'origin_score': score_valid}
        test_data_id = {'x_data': x_id_test, 'y_label': y_test, 'tid': tid_test, 'origin_score': score_test}
        id_corpus = {'train': train_data_id, 'valid':valid_data_id, 'test': test_data_id}

        return id_corpus, max_len_doc, avg_len_doc
    # end def get_id_corpus

    # #
    # def _sent2id(self, sent):
    #     """ return id-converted sentence """

    #     # note that, it is not zero padded yet here
    #     id_sent = []
    #     if self.token_type.startswith("word"):
    #         tokens_sent = nltk.word_tokenize(sent)  # word level tokenizer

    #         id_sent = [self.rev_vocab.get(t, self.unk_id) for t in tokens_sent]
    #         id_sent = [self.bos_id] + id_sent + [self.eos_id]  # add BOS and EOS to make an id-converted sentence
    #     elif self.token_type.startswith("sentpiece"):
    #         tokens_sent = self.tokenizer.tokenize(sent)
    #         #tokens_sent = ["[CLS]"] + tokens_sent + ["[SEP]"]  # CLS is required when we use classification task in BERT
    #         tokens_sent = tokens_sent + ["[SEP]"]  # otherwise, SEP is enough
    #         id_sent = self.tokenizer.convert_tokens_to_ids(tokens_sent)

    #     return id_sent

    # #
    # def _to_id_corpus(self, data):
    #     """ return id-converted corpus"""

    #     results = []
    #     max_len_doc = -1
    #     for cur_doc in data:
    #         temp = []
    #         for raw_sent in cur_doc:
    #             id_sent = self._sent2id(raw_sent)  # convert to id

    #             temp.append(id_sent)
    #         results.append(temp)

    #         flat_doc = [item for sublist in temp for item in sublist]
    #         if len(flat_doc) > max_len_doc:
    #             max_len_doc = len(flat_doc)

    #     return results, max_len_doc

    #
    def read_kfold(self, config):

        """ not used for asap, to follow the same setting with previous work"""

        return

   #
    def generate_kfold(self, config, seed):
        # prepare target directory and file
        path_asap = config.data_dir
        # path_fold_dir = config.data_dir_cv

        #
        train_raw_pd = pd.read_csv(os.path.join(path_asap, "training_set_rel3.tsv"), sep='\t', header=0,
                                          encoding="ISO-8859-1")  # skip the first line
        valid_raw_pd = pd.read_csv(os.path.join(path_asap, "valid_set.tsv"), sep='\t', header=0, encoding="ISO-8859-1")
        test_raw_pd = pd.read_csv(os.path.join(path_asap, "test_set.tsv"), sep='\t', header=0, encoding="ISO-8859-1")

        #
        path_fold_dir = "fold_"  # to keep consistency with previous work, name it "fold_"
        path_asap_cv = os.path.join(path_asap, path_fold_dir)
        col_asap = list(train_raw_pd)

        #
        self._save_kfold(config, path_cv_output, col_data, train_raw_pd)
        self._save_kfold(config, path_cv_output, col_data, valid_raw_pd)
        self._save_kfold(config, path_cv_output, col_data, test_raw_pd)

        return

    # end generate_kfold

    #
    def _save_kfold(self, config, path_asap_cv, col_asap, input_pd):
        ## asap dataset is already provided by train, valid, and test, thus it is chunked manually for cv
        
        # asap consists of 8 prompt, thus it should be chunked with prompt level
        list_prompts_folded = []
        seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        for prompt_id in range(1,9):
            cur_prompt_pd = input_pd.loc[input_pd['essay_set'] == prompt_id]
            cur_prompt_np = cur_prompt_pd.values

            # make chunks to define train, valid, test
            rand_index = np.array(range(len(cur_prompt_np)))
            np.random.shuffle(rand_index)
            shuffled_input = cur_prompt_np[rand_index]
            list_chunk_input = np.array_split(shuffled_input, config.num_fold)

            cur_map_prompt_fold = dict()
            for cur_fold in range(config.num_fold):
                # prepare chunks for train according to each k-fold
                train_chunks = []
                valid_chunks = []
                test_chunks = []
                for cur_ind in range(config.num_fold):
                    cur_test_ind = (config.num_fold - 1) - cur_fold
                    cur_valid_ind = cur_test_ind - 1
                    if cur_valid_ind < 0:
                        cur_valid_ind = cur_valid_ind + config.num_fold

                    if cur_ind == cur_test_ind:
                        test_chunks.append(list_chunk_input[cur_ind])
                    elif cur_ind == cur_valid_ind:
                        valid_chunks.append(list_chunk_input[cur_ind])
                    else:
                        train_chunks.append(list_chunk_input[cur_ind])

                #
                cur_train_np = np.concatenate(train_chunks, axis=0)
                cur_valid_np = np.concatenate(valid_chunks, axis=0)
                cur_test_np = np.concatenate(test_chunks, axis=0)
                cur_map_prompt_fold[cur_fold] = {"train": cur_train_np, "valid": cur_valid_np, "test": cur_test_np}

            list_prompts_folded.append(cur_map_prompt_fold)

        for cur_fold in range(config.num_fold):

            if not os.path.exists(path_asap_cv + str(cur_fold)):
                os.makedirs(path_asap_cv + str(cur_fold))

            cur_fold_train = []
            cur_fold_valid = []
            cur_fold_test = []

            # merge prompts again for cur-fold level
            for prompt_id in range(0,8):
                cur_prompt_train = list_prompts_folded[prompt_id][cur_fold]["train"]
                cur_prompt_valid = list_prompts_folded[prompt_id][cur_fold]["valid"]
                cur_prompt_test = list_prompts_folded[prompt_id][cur_fold]["test"]

                cur_fold_train.append(cur_prompt_train)
                cur_fold_valid.append(cur_prompt_valid)
                cur_fold_test.append(cur_prompt_test)

            fold_train_np = np.concatenate(cur_fold_train, axis=0)
            fold_valid_np = np.concatenate(cur_fold_valid, axis=0)
            fold_test_np = np.concatenate(cur_fold_test, axis=0)

            # finally, save each fold
            pd.DataFrame(fold_train_np, columns=col_asap).to_csv(
                os.path.join(path_asap_cv + str(cur_fold), "train.tsv"), index=None, sep='\t')
            pd.DataFrame(fold_valid_np, columns=col_asap).to_csv(
                os.path.join(path_asap_cv + str(cur_fold), "dev.tsv"), index=None, sep='\t')
            pd.DataFrame(fold_test_np, columns=col_asap).to_csv(
                os.path.join(path_asap_cv + str(cur_fold), "test.tsv"), index=None, sep='\t')

        return

# end class
