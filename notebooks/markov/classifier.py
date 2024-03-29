from .model import Model
from .confusion_matrix import ConfusionMatrix
from .settings import DEF_VOCAB_SIZE, DEF_SMOOTHING_WEIGHT

import math

class Classifier:
    model_pos: Model
    model_neg: Model
    __log_prior_pos__: float
    __log_prior_neg__: float
    __num_positives__: int
    __num_negatives__: int
    order: int

    def __init__(self, order: int, instances: list, labels: list, multithread: bool = False, verbose: bool = False, 
                 vocab_size: int = DEF_VOCAB_SIZE, smoothing_weight: float = DEF_SMOOTHING_WEIGHT):
        
        positives = []
        negatives = []

        if (len(instances) != len(labels)):
            raise ValueError('Instances and labels must have the same length.')
        
        for i in range(len(instances)):
            if labels[i]:
                positives.append(instances[i])
            else:
                negatives.append(instances[i])

        self.model_pos = Model(order, positives, multithread=multithread, verbose=verbose, vocab_size=vocab_size, smoothing_weight=smoothing_weight)
        self.model_neg = Model(order, negatives, multithread=multithread, verbose=verbose, vocab_size=vocab_size, smoothing_weight=smoothing_weight)
        self.__num_positives__ = len(positives)
        self.__num_negatives__ = len(negatives)
        self.reset_prior()
        self.order = order

        if verbose:
            print(f'Classifier trained: {len(positives)} positive, {len(negatives)} negative, {prior_pos * 100:.2f}% positive prior probability.\n(Hint: the prior probability can be modified with the set_prior() method.)')

    def override_prior(self, prior: float):
        self.__log_prior_pos__ = math.log(prior)
        self.__log_prior_neg__ = math.log(1 - prior)

    def reset_prior(self):
        prior_pos = self.__num_positives__ / (self.__num_positives__ + self.__num_negatives__)
        self.__log_prior_pos__ = math.log(prior_pos)
        self.__log_prior_neg__ = math.log(1 - prior_pos)

    def add_instance(self, seq, label, update_prior: bool = True):
        if label:
            self.model_pos.add_instance(seq)
            self.__num_positives__ += 1
        else:
            self.model_neg.add_instance(seq)
            self.__num_negatives__ += 1

        if update_prior:
            self.reset_prior()

    def classify(self, seq) -> bool:
        return self.model_pos.get_log_sum_prob(seq) + self.__log_prior_pos__ > self.model_neg.get_log_sum_prob(seq) + self.__log_prior_neg__

    def eval(self, instances: list, labels: list) -> ConfusionMatrix:

        res = ConfusionMatrix()
        
        for i in range(len(instances)):
            if self.classify(instances[i]):
                if labels[i]:
                    res.tp += 1
                else:
                    res.fp += 1
            else:
                if labels[i]:
                    res.fn += 1
                else:
                    res.tn += 1
        
        return res
    