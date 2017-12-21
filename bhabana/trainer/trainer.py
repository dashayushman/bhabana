from abc import ABC
from abc import abstractmethod


class Trainer(ABC):

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def train(self, batch):
        pass

    @abstractmethod
    def evaluate(self, batch):
        pass

    @abstractmethod
    def save(self, epoch, global_step):
        pass

    @abstractmethod
    def log(self, batch):
        pass

    @abstractmethod
    def loss_has_improved(self, loss):
        pass