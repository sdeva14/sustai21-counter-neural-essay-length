import logging
import torch
import numpy as np

from evaluators.eval_base import Eval_Base
from evaluators.my_kappa_calculator import quadratic_weighted_kappa

from sklearn.metrics import cohen_kappa_score as kappa

class Eval_Qwk(Eval_Base):
    """ evaluation class for quadratic weighted Kappa """

    logger = logging.getLogger(__name__)

    #
    def __init__(self, config, min_rating, max_rating, corpus_target):
        super(Eval_Qwk, self).__init__(config)

        self.prompt_id_train = config.essay_prompt_id_train
        self.prompt_id_test = config.essay_prompt_id_test

        self.min_rating = min_rating
        self.max_rating = max_rating
        self.use_gpu = config.use_gpu

        self.output_size = config.output_size
        self.corpus_target = config.corpus_target
        self.loss_type = config.loss_type.lower()

        self.pred_list = []  # list of Tensor
        self.origin_label_list = []  # list of float
        self.tid_list = []

        self.score_ranges = {
            0: (0, 60), 1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60)
        }
        #self.relabel_ranges = corpus_target.relabel_ranges

        self.map_suppl = {}  # storage for supplementary data

        return

    #
    def _convert_to_dataset_friendly_scores(self, scores_np):
        """ revert to original scale, which was scaled for loss function """

        scores_np = scores_np * (self.max_rating - self.min_rating) + self.min_rating

        return scores_np

    #
    def eval_update(self, model_output, label_y, tid, origin_label=None):
        """ store eval data in every step """

        if self.output_size > 1:
            _, model_output = torch.max(model_output, 1)  # model_output: (batch_size, num_class_out)

        # update prediction list
        self.pred_list.append(model_output)
        # self.pred_list = torch.stack(model_output, dim=0)

        if origin_label is not None:
            origin_label = origin_label.squeeze().tolist()
            self.origin_label_list = self.origin_label_list + origin_label

        list_tid = tid.squeeze().tolist()
        if not isinstance(list_tid, list): list_tid = [list_tid]
        self.tid_list = self.tid_list + list_tid

        return

    #
    def eval_measure(self, is_test):
        """ return evaluation performance """

        # cur_pred_list = torch.stack(self.pred_list, dim=0)
        # cur_pred_list = cur_pred_list.view(-1)
        # cur_pred_list = torch.cat(self.pred_list, dim=0).squeeze(1)
        cur_pred_list = torch.cat(self.pred_list, dim=0)

        pred_list_np = None
        if self.use_gpu:    pred_list_np = cur_pred_list.cpu().numpy()
        else:   pred_list_np = cur_pred_list.numpy()

        if self.loss_type == "mseloss":
            self.rescaled_pred = self._convert_to_dataset_friendly_scores(pred_list_np)
            self.rescaled_pred = np.rint(self.rescaled_pred)
            self.rescaled_pred = np.minimum(self.rescaled_pred, self.max_rating)
            self.rescaled_pred = self.rescaled_pred.astype('int32')
        else:
            self.rescaled_pred = pred_list_np

        # if self.corpus_target == "asap":
        #     self.rescaled_pred = self._convert_to_dataset_friendly_scores(pred_list_np)
        #     self.rescaled_pred = np.rint(self.rescaled_pred).astype('int32')
        # else:
        #     self.rescaled_pred = pred_list_np
        
        self.origin_label_np = np.array(self.origin_label_list).astype('int32')

        # print(self.rescaled_pred)
        # print(self.origin_label_np)

        # #
        # qwk = quadratic_weighted_kappa(self.rescaled_pred, self.origin_label_np, self.min_rating, self.max_rating)

        qwk = kappa(self.rescaled_pred, self.origin_label_np, weights="quadratic")  # sklearn library
        
        if is_test:
            self.eval_history.append(qwk)

        #
        self.eval_reset()

        return qwk

    #
    def eval_reset(self):
        self.pred_list = []
        self.origin_label_list = []  # list of float
        self.tid_list = []

        # self.map_suppl = {}

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
