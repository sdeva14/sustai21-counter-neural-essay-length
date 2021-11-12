# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils import INT, FLOAT, LONG

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import logging
logger = logging.getLogger()

# from apex import amp
#from parallel import DataParallelModel, DataParallelCriterion  # parallel huggingface

#
def get_loss_func(config, pad_id=None):
	loss_func = None
	if config.loss_type.lower() == 'crossentropyloss':
		print("Use CrossEntropyLoss")
		loss_func = nn.CrossEntropyLoss(ignore_index=pad_id)
	elif config.loss_type.lower() == 'nllloss':
		print("Use NLLLoss")
		loss_func = nn.NLLLoss(ignore_index=pad_id)
	elif config.loss_type.lower() == 'multilabelsoftmarginloss':
		print("MultiLabelSoftMarginLoss")
		loss_func = nn.MultiLabelSoftMarginLoss()
	elif config.loss_type.lower() == 'mseloss':
		print("MSELoss")
		loss_func = nn.MSELoss()

	return loss_func

# end get_loss_func

#
def validate(model, evaluator, dataset_test, config, loss_func, is_test=False):
	model.eval()
	losses = []

	sampler_test = SequentialSampler(dataset_test) if config.local_rank == -1 else DistributedSampler(dataset_test)
	dataloader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=config.batch_size)

	for text_inputs, label_y, *remains in dataloader_test:
		mask_input = remains[0]
		len_seq = remains[1]
		len_sents = remains[2]
		tid = remains[3]
		cur_origin_score = remains[-1]  # it might not be origin score when it is not needed for the dataset, then will be ignored

		text_inputs = utils.cast_type(text_inputs, LONG, config.use_gpu)
		mask_input = utils.cast_type(mask_input, FLOAT, config.use_gpu)
		len_seq = utils.cast_type(len_seq, FLOAT, config.use_gpu)

		with torch.no_grad():
			coh_score = model(text_inputs=text_inputs, mask_input=mask_input, len_seq=len_seq, len_sents=len_sents, tid=tid, mode="")  # model.forward; now it returns the loss

			if config.output_size == 1:
				coh_score = coh_score.view(text_inputs.shape[0])
			else:
				coh_score = coh_score.view(text_inputs.shape[0], -1)

			if config.loss_type.lower() == 'mseloss':
				label_y = utils.cast_type(label_y, FLOAT, config.use_gpu)
			else:
				label_y = utils.cast_type(label_y, LONG, config.use_gpu)
			label_y = label_y.view(text_inputs.shape[0])

			if loss_func is not None:
				loss = loss_func(coh_score, label_y)

				losses.append(loss.item())

			evaluator.eval_update(coh_score, label_y, tid, cur_origin_score)
			# evaluator.eval_update(model_score_eval, label_y, tid, cur_origin_score)
		# end with torch.no_grad()
	# end for batch_num

	eval_measure = evaluator.eval_measure(is_test)
	eval_best_val = None
	if is_test:
		eval_best_val = max(evaluator.eval_history)

	if loss_func is not None:
		valid_loss = sum(losses) / len(losses)
		if is_test:
			logger.info("Total valid loss {}".format(valid_loss))
	else:
		valid_loss = np.inf
	
	if is_test:
		logger.info("{} on Test {}".format(evaluator.eval_type, eval_measure))
		logger.info("Best {} on Test {}".format(evaluator.eval_type, eval_best_val))
	else:
		logger.info("{} on Valid {}".format(evaluator.eval_type, eval_measure))

	return valid_loss, eval_measure, eval_best_val
# end validate

#
def train(model, optimizer, scheduler, dataset_train, dataset_valid, dataset_test, config, evaluator):  # valid_feed can be None

	patience = 10  # wait for at least 10 epoch before stop
	valid_loss_threshold = np.inf
	best_valid_loss = np.inf
	best_eval_valid = 0.0
	final_eval_best = 0.0

	sampler_train = RandomSampler(dataset_train) if config.local_rank == -1 else DistributedSampler(dataset_train)
	dataloader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=config.batch_size)

	batch_cnt = 0
	ckpt_step = len(dataloader_train.dataset) // dataloader_train.batch_size
	logger.info("**** Training Begins ****")
	logger.info("**** Epoch 0/{} ****".format(config.max_epoch))

	loss_func = None
	# if config.use_parallel and not config.use_apex:
	# if config.n_gpu > 1 and not config.use_apex:
	if config.n_gpu > 1 and config.local_rank == -1:
		loss_func = get_loss_func(config=config, pad_id=model.module.pad_id)
	else:
		loss_func = get_loss_func(config=config, pad_id=model.pad_id)

	if config.use_gpu:
		loss_func.cuda()

	# epoch loop
	model.train()
	for cur_epoch in range(config.max_epoch):

		# loop until traverse all batches
		train_loss = []
		for text_inputs, label_y, *remains in dataloader_train:
			mask_input = remains[0]
			len_seq = remains[1]
			len_sents = remains[2]
			tid = remains[3]

			text_inputs = utils.cast_type(text_inputs, LONG, config.use_gpu)
			mask_input = utils.cast_type(mask_input, FLOAT, config.use_gpu)
			len_seq = utils.cast_type(len_seq, FLOAT, config.use_gpu)

			# training for this batch
			optimizer.zero_grad()
			
			coh_score = model(text_inputs=text_inputs, mask_input=mask_input, len_seq=len_seq, len_sents=len_sents, tid=tid, mode="")  # model.forward; now it returns the loss
			
			if config.output_size == 1:
				coh_score = coh_score.view(text_inputs.shape[0])
			else:
				coh_score = coh_score.view(text_inputs.shape[0], -1)


#           # get loss
			if config.loss_type.lower() == 'mseloss':
				label_y = utils.cast_type(label_y, FLOAT, config.use_gpu)
			else:
				label_y = utils.cast_type(label_y, LONG, config.use_gpu)
			label_y = label_y.view(text_inputs.shape[0])

			loss = loss_func(coh_score, label_y)
			if config.n_gpu > 1:
				loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

			loss.backward()
			# with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
			#     scaled_loss.backward()
			
			torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
			#for p in model.parameters():  # gradient control manually
			#    if p.grad is not None:
			#        p.data.add_(-config.init_lr, p.grad.data)
			# clip_grad_norm_(amp.master_params(optimizer), config.clip_n2)

			# update optimizer and scheduler
			optimizer.step()
			if scheduler is not None:
				# scheduler.step()
				scheduler.step(loss)

			# for param_group in optimizer.param_groups:
			# 	print(param_group['lr'])

			train_loss.append(loss.item())

			# temporal averaging for encoders (proposed in ICLR18)
			if config.encoder_type == "reg_lstm" and config.beta_ema > 0:
				if config.n_gpu > 1:    model.module.encoder_coh.update_ema()
				else:   model.encoder_coh.update_ema()

			batch_cnt = batch_cnt + 1

			# print train process
			if batch_cnt % config.print_step == 0:
				logger.info("{}/{}-({:.3f})".format(batch_cnt % config.ckpt_step,
																		 config.ckpt_step,
																		 loss))

			## validation
			if batch_cnt % ckpt_step == 0:  # manual epoch printing
			# if i == batch_num-1:  # every epoch
				logger.info("\n=== Evaluating Model ===")

				# validation
				eval_cur_valid = -1
				if dataset_valid is not None:
					loss_valid, eval_cur_valid, _ = validate(model, evaluator, dataset_valid, config, loss_func)
					logger.info("")

				if eval_cur_valid >= best_eval_valid or dataset_valid is None:
				# if dataset_valid is not None:
					logger.info("Best {} on Valid {}".format(evaluator.eval_type, eval_cur_valid))
					best_eval_valid = eval_cur_valid

					valid_loss, eval_last, eval_best = validate(model, evaluator, dataset_test, config, loss_func, is_test=True)
					if eval_best > final_eval_best: 
						final_eval_best = eval_best

						# save model
						if config.save_model:
							logger.info("Model Saved.")
							torch.save(model.state_dict(), os.path.join(config.session_dir, "model"))

						# save prediction log for error analysis
						if config.gen_logs:
							pred_log_name = "log_pred_" + str(config.essay_prompt_id_train) + "_" + str(config.essay_prompt_id_test) + "_" + str(config.cur_fold) + ".log"
							if config.eval_type.lower() == "qwk":
								pred_out = np.stack((evaluator.rescaled_pred, evaluator.origin_label_np, evaluator.tid_np))
								np.savetxt(os.path.join(config.session_dir, pred_log_name), pred_out, fmt ='%.0f')
							elif config.eval_type.lower() == "accuracy":
								pred_out = np.stack((evaluator.pred_list_np, evaluator.origin_label_np, evaluator.tid_np))
								pred_out = pred_out.T
								np.savetxt(os.path.join(config.session_dir, pred_log_name), pred_out, fmt ='%.0f')


						# # error analysis: std data for lexical cohesion
						# if config.gen_logs and config.target_model == "ilcr_scd":
						# 	std_log_name = "log_std_" + str(config.essay_prompt_id_train) + "_" + str(config.essay_prompt_id_test) + "_" + str(config.cur_fold) + ".log"
						# 	# # file read
						# 	std_data = evaluator.map_suppl["std"]
						# 	with open(os.path.join(config.session_dir, std_log_name), "w") as f:
						# 		f.write(repr(std_data))

				evaluator.map_suppl={}  # reset

				# early stopping parts (disabled)
				# if valid_loss < best_valid_loss:
				#     if valid_loss <= valid_loss_threshold * config.improve_threshold:
				#         patience = max(patience,
				#                        cur_epoch * config.patient_increase)
				#         valid_loss_threshold = valid_loss
				#         logger.info("Update patience to {}".format(patience))
				#     # end if if valid_loss <= valid_loss_threshold * config.improve_threshold
				#
				#     best_valid_loss = valid_loss
				# # end if valid_loss < best_valid_loss:

				# if cur_epoch >= config.max_epoch \
				#         or config.early_stop and patience <= cur_epoch:
				#     if cur_epoch < config.max_epoch:
				#         logger.info("!!Early stop due to run out of patience!!")
				#
				#     logger.info("Best validation loss %f" % best_valid_loss)
				#
				#     return
				# end if if cur_epoch >= config.max_epoch \

				# exit eval model
				model.train()
				train_loss = []
				logger.info("\n**** Epcoch {}/{} ****".format(cur_epoch,
															  config.max_epoch))
			# end valdation

			if config.use_gpu and config.empty_cache:
				torch.cuda.empty_cache()    # due to memory shortage
		# end batch loop
	# end epoch loop
	logger.info("Best {} on Test {}".format(evaluator.eval_type, final_eval_best))
	logger.info("")

	return final_eval_best
# end train




