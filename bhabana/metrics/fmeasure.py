import numpy as np

from bhabana.metrics import Metric
from sklearn.metrics import f1_score


class FMeasure(Metric):

    def __call__(self, pred, gt):
        return self.calculate(pred, gt)

    def calculate(self, pred, gt):
        return f1_score(np.argmax(gt, axis=1), np.argmax(pred, axis=1),
                        average="weighted")