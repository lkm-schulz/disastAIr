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
        return f'Total: {total:.1f}, TP: {self.tp:.1f}, TN: {self.tn:.1f}, FP: {self.fp:.1f}, FN: {self.fn:.1f}, Acc: {self.get_accuracy():.3f}, Prc: {self.get_precision():.3f}, Rec: {self.get_recall():.3f}, F1: {self.get_f1():.3f}'
    
    def get_f1(self):
        prec = self.get_precision()
        rec = self.get_recall()
        return (2 * ((prec * rec) / (prec + rec))) if prec + rec > 0 else 0
    
    def get_precision(self):
        return self.tp / (self.tp + self.fp) if self.tp > 0 else 0

    def get_recall(self):
        return self.tp / (self.tp + self.fn) if self.tp > 0 else 0
    
    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def __add__(self, rhs: 'ConfusionMatrix') -> 'ConfusionMatrix':
        tp = self.tp + rhs.tp
        fp = self.fp + rhs.fp
        tn = self.tn + rhs.tn
        fn = self.fn + rhs.fn
        return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)

    def __truediv__(self, divisor) -> 'ConfusionMatrix':
        tp = self.tp / divisor
        fp = self.fp / divisor
        tn = self.tn / divisor
        fn = self.fn / divisor
        return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)

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
        