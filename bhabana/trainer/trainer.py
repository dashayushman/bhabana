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
    def save(self):
        pass

    @abstractmethod
    def loss_has_improved(self):
        pass

    @abstractmethod
    def update_loss_history(self, loss):
        pass

    @abstractmethod
    def log(self, name, value):
        pass

    @abstractmethod
    def closure(self):
        pass

    def time_to_evaluate(self, evaluate_every, current_step):
        if current_step % evaluate_every == 0 and current_step != 0:
            return True
        else:
            return False

    def time_to_save(self, save_every, current_step):
        return self.time_to_evaluate(save_every, current_step)