from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int

    def __init__(self, tp: int = 0, tn: int = 0, fp: int = 0, fn: int = 0):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def __str__(self):
        total = self.tp + self.fp + self.tn + self.fn
        return f'Total: {total:.1f}, TP: {self.tp:.1f}, TN: {self.tn:.1f}, FP: {self.fp:.1f}, FN: {self.fn:.1f}, Prc: {self.get_precision():.3f}, Rec. {self.get_recall():.3f}, F1: {self.get_f1():.3f}'
    
    def get_f1(self):
        return 2 * ((self.get_precision() * self.get_recall()) / (self.get_precision() + self.get_recall()))
    
    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        return self.tp / (self.tp + self.fn) 

    def add(self, rhs: 'ConfusionMatrix'):
        self.tp += rhs.tp
        self.fp += rhs.fp
        self.tn += rhs.tn
        self.fn += rhs.fn

    def div(self, divisor):
        self.tp /= divisor
        self.fp /= divisor
        self.tn /= divisor
        self.fn /= divisor

    def get_display(self, cmap=plt.cm.Blues, display_labels=["Positive", "Negative"]):
        actual = ([1] * round(self.tp)) + ([0] * round(self.fp)) + ([0] * round(self.tn)) + ([1] * round(self.fn))
        pred = ([1] * round(self.tp)) + ([1] * round(self.fp)) + ([0] * round(self.tn)) + ([0] * round(self.fn))
        return ConfusionMatrixDisplay.from_predictions(y_pred=pred, y_true=actual, cmap=cmap, display_labels=display_labels)

    @classmethod
    def average(cls, matrices: list['ConfusionMatrix']):
        result = ConfusionMatrix()
        for cm in matrices:
            result.add(cm)
        result.div(len(matrices))

        return result
        