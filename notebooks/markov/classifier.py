from .model import Model
from .confusion_matrix import ConfusionMatrix
from .settings import DEF_VOCAB_SIZE, DEF_SMOOTHING_WEIGHT

import math

class Classifier:
    model_pos: Model
    model_neg: Model
    __log_prior_pos__: float
    __log_prior_neg__: float
    order: int

    def __init__(self, order: int, positives: list, negatives: list, verbose: bool = False, 
                 vocab_size: int = DEF_VOCAB_SIZE, smoothing_weight: float = DEF_SMOOTHING_WEIGHT):
        
        self.model_pos = Model(order, positives, multithread=True, verbose=verbose, vocab_size=vocab_size, smoothing_weight=smoothing_weight)
        self.model_neg = Model(order, negatives, multithread=True, verbose=verbose, vocab_size=vocab_size, smoothing_weight=smoothing_weight)
        self.__log_prior_pos__ = math.log(len(positives)) - math.log((len(positives) + len(negatives)))
        self.__log_prior_neg__ = math.log(len(negatives)) - math.log((len(positives) + len(negatives)))
        self.order = order

    def classify(self, seq) -> bool:
        return self.model_pos.get_log_sum_prob(seq) + self.__log_prior_neg__ > self.model_neg.get_log_sum_prob(seq) + self.__log_prior_neg__

    def eval(self, positives: list, negatives: list) -> ConfusionMatrix:

        res = ConfusionMatrix()

        for instance in positives:
            if self.classify(instance):
                res.tp += 1
            else:
                res.fn += 1

        for instance in negatives:
            if not self.classify(instance):
                res.tn += 1
            else:
                res.fp += 1
        
        return res
    