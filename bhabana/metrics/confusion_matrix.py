import numpy as np

from bhabana.metrics import Metric
from sklearn.metrics import classification_report


class ClassificationReport(Metric):

    def __call__(self, pred, gt):
        return self.calculate(pred, gt)

    def calculate(self, pred, gt):
        return classification_report(np.argmax(gt, axis=1), np.argmax(pred, axis=1))