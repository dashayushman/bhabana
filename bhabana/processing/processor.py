from abc import ABC
from abc import abstractmethod


class Processor(ABC):

    @abstractmethod
    def is_valid_data(self, data):
        pass

    @abstractmethod
    def process(self, data):
        pass
