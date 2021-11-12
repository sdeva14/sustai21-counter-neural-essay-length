import logging
import torch
import numpy as np

from evaluators.eval_base import Eval_Base

import utils
from utils import INT, FLOAT, LONG

class Eval_Acc(Eval_Base):
    """ evaluation class for accuracy """

    logger = logging.getLogger(__name__)

    #
    def __init__(self, config):
        super(Eval_Acc, self).__init__(config)

        self.correct = 0.0
        self.total = 0.0

        self.use_gpu = config.use_gpu

        self.gcdc_ranges = (1, 3)  # used if it needs to re-scale
        self.pred_list = []
        self.label_list = []
        self.tid_list = []

        self.map_suppl = {}  # storage for supplementary data

        return

    #
    def _convert_to_origin_scale(self, scores):
        """ need to revert to original scale which is scaled for loss function """
        min_rating, max_rating = self.gcdc_ranges
        scores = scores * (max_rating - min_rating) + min_rating

        return scores

    #
    def eval_update(self, model_output, label_y, tid, origin_label=None):
        """ update data in every step for evalaution """

        _, predicted = torch.max(model_output, 1)  # model_output: (batch_size, num_class_out)
        self.correct += (predicted == label_y).sum().item()
        self.total += predicted.size(0)

        # self.pred_list.append(model_output)
        # self.label_list.append(origin_label)

        list_predict = predicted.squeeze().tolist()
        list_label = label_y.squeeze().tolist()
        # list_label = origin_label.squeeze().tolist()
        list_tid = list(tid)
        if not isinstance(list_predict, list): list_predict = [list_predict]
        if not isinstance(list_label, list): list_label = [list_label]
        if not isinstance(list_tid, list): list_tid = [list_tid]

        self.pred_list = self.pred_list + list_predict
        self.label_list = self.label_list + list_label
        self.tid_list = self.tid_list + list_tid


        return

    #
    def eval_update_mse(self, model_output, label_y, origin_label=None):
        # get accuracy

        self.pred_list.append(model_output)
        self.label_list.append(origin_label)

        return

    #
    def eval_measure(self, is_test):
        """ calculate evaluation from stored data """

        # get acc
        accuracy = self.correct / self.total

        # for err analysis
        # cur_pred_list = torch.cat(self.pred_list, dim=0)
        # cur_label_list = torch.cat(self.label_list, dim=0)

        # print(self.pred_list)
        # print(self.label_list)

        # self.pred_list_np = np.array(self.origin_label_list).astype('int32')
        # self.origin_label_np = np.array(self.origin_label_list).astype('int32')
        self.pred_list_np = self.pred_list
        self.origin_label_np = self.label_list
        self.tid_np = self.tid_list

        # self.pred_list_np = None
        # if self.use_gpu:    
        #     self.pred_list_np = cur_pred_list.cpu().numpy()
        #     self.origin_label_np = cur_pred_list.cpu().numpy()
        # else:   
        #     self.pred_list_np = cur_label_list.cpu().numpy()
        #     self.origin_label_np = cur_label_list.numpy()
        

        # reset
        self.eval_reset()

        # store performance for test mode
        if is_test:
            self.eval_history.append(accuracy)


        return accuracy

    #
    def eval_measure_mse(self):
        #
        cur_pred_list = torch.cat(self.pred_list, dim=0)
        reverted_pred = self._convert_to_origin_scale(cur_pred_list)
        reverted_pred = torch.round(reverted_pred)
        predicted = utils.cast_type(reverted_pred, LONG, self.use_gpu)

        cur_label_list = [label for row in self.label_list for label in row]
        cur_label_list = torch.LongTensor(cur_label_list)
        cur_label_list = utils.cast_type(cur_label_list, LONG, self.use_gpu)
        cur_label_list = cur_label_list.view(cur_label_list.shape[0], 1)

        correct = (predicted == cur_label_list).sum().item()
        total = predicted.size(0)
        accuracy = correct / total

        self.eval_reset()

        return accuracy

    #
    def eval_reset(self):
        self.correct = 0.0
        self.total = 0.0

        self.pred_list = []
        self.label_list = []  # list of float
        self.tid_list = []

        return

    #
    def save_suppl(self, name, supp_data):
        # supp_data = supp_data.squeeze().tolist()  # torch -> list
        if name not in self.map_suppl:
            self.map_suppl[name] = supp_data
        else:
            stored = self.map_suppl[name]
            updated = stored + supp_data
            self.map_suppl[name] = updated

        return