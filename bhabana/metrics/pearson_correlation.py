from bhabana.metrics import Metric
from scipy.stats import pearsonr


class PearsonCorrelation(Metric):

    def __call__(self, pred, gt):
        return self.calculate(pred, gt)

    def calculate(self, pred, gt):
        return pearsonr(pred, gt)