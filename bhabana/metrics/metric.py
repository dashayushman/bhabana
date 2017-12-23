from abc import ABC
from abc import abstractmethod


class Metric(ABC):

    @abstractmethod
    def calculate(self, pred, gt):
        pass