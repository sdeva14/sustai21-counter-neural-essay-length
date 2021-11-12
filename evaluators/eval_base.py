import logging

class Eval_Base(object):
    logger = logging.getLogger(__name__)

    #
    def __init__(self, config):
        super(Eval_Base, self).__init__()
        self.eval_type = config.eval_type  # accuracy, qwk,
        
        self.eval_history = []

        return

    #
    def eval_update(self, model_output, label_y, origin_label=None):
        raise NotImplementedError
    #
    def eval_measure(self):
        raise NotImplementedError
    #
    def eval_reset(self):
        raise NotImplementedError
