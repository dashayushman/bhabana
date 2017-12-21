import numbers

from bhabana.processing import Processor
from bhabana.utils.generic_utils import to_categorical


class OneHot(Processor):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def is_valid_data(self, data):
        validation_message = {"is_valid": True}
        if type(data) != int:
            validation_message["is_valid"] = False
            validation_message["error"] = "The Class ID must be an Integer. " \
                                      "Currently it is {}".format(type(data))
        return validation_message

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        return to_categorical(data, self.n_classes).tolist()
