# -*- coding: utf-8 -*-

#
import os
import argparse
import logging
import time

#
import numpy as np
import torch
import torch.nn as nn

# written codes
import build_config
import utils

import corpus.corpus_asap
import corpus.corpus_toefl

import w2vEmbReader

from models.optim_hugging import AdamW, WarmupLinearSchedule, WarmupCosineSchedule

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import training

from evaluators import eval_acc, eval_qwk

from models.model_CoNLL17_Essay import Model_CoNLL17_Essay
from models.model_EMNLP18_Centt import Model_EMNLP18_Centt
import models.model_SkipFlow as model_skipFlow

import models.model_ILCR_Avg as model_ILCR_avg
from models.model_ILCR_KLD import Coh_Model_ILCR_KLD

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from corpus.dataset_asap import Dataset_ASAP
from corpus.dataset_toefl import Dataset_TOEFL

########################################################

# global parser for arguments
parser = argparse.ArgumentParser()
arg_lists = []

###########################################
###########################################

#
def get_w2v_emb(config, corpus_target):
    embReader = w2vEmbReader.W2VEmbReader(config=config, corpus_target=corpus_target)
    
    return embReader                                              

#
def get_corpus_target(config):
    corpus_target = None
    logger = logging.getLogger()

    if config.corpus_target.lower() == "asap":
        logger.info("Corpus: ASAP")
        corpus_target = corpus.corpus_asap.CorpusASAP(config)
    elif config.corpus_target.lower() == "toefl":
        logger.info("Corpus: TOEFL")
        corpus_target = corpus.corpus_toefl.CorpusTOEFL(config)

    return corpus_target
# end get_corpus_target


#
def get_dataset(config, id_corpus, pad_id):
    dataloader_train = None
    dataloader_valid = None
    dataloader_test = None

    if config.corpus_target.lower() == "asap":
        dataset_train = Dataset_ASAP(id_corpus["train"], config, pad_id)
        dataset_valid = Dataset_ASAP(id_corpus["valid"], config, pad_id)
        dataset_test = Dataset_ASAP(id_corpus["test"], config, pad_id)
    elif config.corpus_target.lower() == "toefl":
        dataset_train = Dataset_TOEFL(id_corpus["train"], config, pad_id)
        dataset_valid = Dataset_TOEFL(id_corpus["valid"], config, pad_id)
        dataset_test = Dataset_TOEFL(id_corpus["test"], config, pad_id)


    return dataset_train, dataset_valid, dataset_test

#
def get_model_target(config, corpus_target, embReader):
    model = None
    logger = logging.getLogger()

    if config.target_model.lower().startswith("emnlp18"):
        logger.info("Model: EMNLP18")
        model = Model_EMNLP18_Centt(config=config, corpus_target=corpus_target, embReader=embReader)
    elif config.target_model.lower().startswith("conll17"):
        logger.info("Model: CoNLL17")
        model = Model_CoNLL17_Essay(config=config, corpus_target=corpus_target, embReader=embReader)
    elif config.target_model.lower().startswith("aaai18"):
        logger.info("Model: AAAI18")
        model = model_skipFlow.Coh_Model_AAAI18(config=config, corpus_target=corpus_target, embReader=embReader)

    elif config.target_model.lower() == "ilcr_avg":
        logger.info("Model: ILCR_Avg")
        model = model_ILCR_avg.Coh_Model_ILCR_Avg(config=config, corpus_target=corpus_target, embReader=embReader)
    elif config.target_model.lower() == "ilcr_kld":
        logger.info("Model: ILCR_KLD")
        model = Coh_Model_ILCR_KLD(config=config, corpus_target=corpus_target, embReader=embReader)


    return model

#
def get_optimizer(config, model, len_trainset):
    # basic style
    model_opt = model.module if hasattr(model, 'module') else model  # take care of parallel
    optimizer = model_opt.get_optimizer(config)

    optimizer = model.get_optimizer(config)
    scheduler = None

    return optimizer, scheduler

    
#
def exp_model(config):
    ## Pre-processing

    # read corpus then generate id-sequence vector
    corpus_target = get_corpus_target(config)  # get corpus class
    corpus_target.read_kfold(config)

    # get embedding class
    embReader = get_w2v_emb(config, corpus_target)

    # update config depending on environment
    config.max_num_sents = corpus_target.max_num_sents  # the maximum number of sentences in document (i.e., document length)
    config.max_len_sent = corpus_target.max_len_sent  # the maximum length of sentence (the number of words)
    # config.max_len_doc = corpus_target.max_len_doc  # the maximum length of document (the number of words)

    # convert to id-sequence for given k-fold
    cur_fold = config.cur_fold
    # id_corpus, max_len_doc, avg_len_doc = corpus_target.get_id_corpus(cur_fold)
    id_corpus, max_len_doc, avg_len_doc = corpus_target.get_id_corpus(cur_fold)
    config.max_len_doc = max_len_doc
    config.avg_len_doc = avg_len_doc
    
    ## get avg of entropy, kld
    # config.avg_ent = corpus_target.get_avg_entropy()
    config.p_map, config.avg_kld = corpus_target.get_p_kld()
    config.map_kld_essays = corpus_target.get_word_kld()

    ## Model
    # prepare batch form
    # batch_data_train, batch_data_valid, batch_data_test = get_batch_loader(config, id_corpus)  # get batch-loader class
    dataset_train, dataset_valid, dataset_test = get_dataset(config, id_corpus, embReader.pad_id)

    #### prepare model
    if torch.cuda.is_available():   config.use_gpu = True
    else: config.use_gpu = False
    model = get_model_target(config, corpus_target, embReader)  # get model class
    optimizer, scheduler = get_optimizer(config, model, len(dataset_train))
    
    device = "cuda"
    if config.local_rank == -1 or not config.use_gpu:  # when it is not distributed mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.n_gpu = torch.cuda.device_count()  # will be 1 or 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # if config.use_parallel:
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", config.local_rank)
        # torch.distributed.init_process_group(backend='nccl')
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        config.world_size = torch.distributed.get_world_size()
        # config.n_gpu = 1

    if config.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
        #                                                  output_device=config.local_rank,
        #                                                  find_unused_parameters=True)
        # model = apex.parallel.DistributedDataParallel(model)
        # config.n_gpu = 1
    elif config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        optimizer = model.module.get_optimizer(config)

    if config.use_gpu:
        model.to(device)
    
    #### run training and evaluation
    min_rating=None
    max_rating=None
    # if config.loss_type.lower() == "mseloss":
    min_rating, max_rating = corpus_target.score_ranges[corpus_target.prompt_id_test]  # in case of MSELoss

    evaluator = None
    if config.eval_type.lower() == "accuracy":
            evaluator = eval_acc.Eval_Acc(config)
    if config.eval_type.lower() == "qwk":
            #min_rating, max_rating = corpus_target.score_ranges[corpus_target.cur_prompt_id]  # in case of asap corpus
            evaluator = eval_qwk.Eval_Qwk(config, min_rating, max_rating, corpus_target) 


    # training
    final_eval_best = training.train(model,
                    optimizer,
                    scheduler,
                    dataset_train=dataset_train,
                    dataset_valid=dataset_valid,
                    dataset_test=dataset_test,
                    config=config,
                    evaluator=evaluator)

    return final_eval_best
# end exp_model

###################################################

if __name__=='__main__':
    ## prepare config
    build_config.process_config()
    config, _ = build_config.get_config()
    utils.prepare_dirs_loggers(config, os.path.basename(__file__))
    logger = logging.getLogger() 

    ## option configure
    # change pad level to sent if EMNLP18, because of sentence level handling
    if config.target_model.lower() == "emnlp18" \
        or config.target_model.lower() == "ilcr_doc_stru":
        config.pad_level = "sent"

    # automatically extract target corpus from dataset path
    if len(config.corpus_target) == 0:
        cur_corpus_name = os.path.basename(os.path.normpath(config.data_dir))
        config.corpus_target = cur_corpus_name

    # domain information for printing
    cur_domain_train = None
    cur_domain_test = None
    if config.corpus_target.lower() == "asap" or config.corpus_target.lower() == "toefl":
        cur_domain_train = config.essay_prompt_id_train
        cur_domain_test = config.essay_prompt_id_test

    ## Run model
    list_cv_attempts=[]
    target_attempts = config.cv_attempts
    
    if config.cur_fold > -1:  # test for specific fold
        if cur_domain_train is not None:
            logger.info("Source domain: {}, Target domain: {}, Cur_fold {}".format(cur_domain_train, cur_domain_test, config.cur_fold))
        eval_best_fold = exp_model(config)
        logger.info("{}-fold eval {}".format(config.cur_fold, eval_best_fold))
    else:
        for cur_attempt in range(target_attempts):  # CV only works when whole k-fold eval mode

            ##
            logger.info("Whole k-fold eval mode")
            list_eval_fold = []
            for cur_fold in range(config.num_fold):
                config.cur_fold = cur_fold
                if cur_domain_train is not None:
                    logger.info("Source domain: {}, Target domain: {}, Cur_fold {}".format(cur_domain_train, cur_domain_test, config.cur_fold))
                cur_eval_best_fold = exp_model(config)
                list_eval_fold.append(cur_eval_best_fold)
            
            avg_cv_eval = sum(list_eval_fold) / float(len(list_eval_fold))
            logger.info("Final k-fold eval {}".format(avg_cv_eval))
            logger.info(list_eval_fold)

            list_cv_attempts.append(avg_cv_eval)

    #
    if target_attempts > 1 and len(list_cv_attempts) > 0:
        avg_cv_attempt = sum(list_cv_attempts) / float(len(list_cv_attempts))
        logger.info("Final CV exp result {}".format(avg_cv_attempt))
        logger.info(list_cv_attempts)

        for cur_score in list_cv_attempts:
            print(cur_score)


