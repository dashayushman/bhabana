import numbers

from bhabana.processing import Processor
from bhabana.utils.data_utils import rescale
from bhabana.utils.data_utils import validate_rescale


class Rescale(Processor):
    def __init__(self, original_lower_bound, original_upper_bound,
                 new_lower_bound, new_upper_bound):
        self.original_range = [original_lower_bound, original_upper_bound]
        self.new_range = [new_lower_bound, new_upper_bound]
        validate_rescale(self.original_range)
        validate_rescale(self.new_range)

    def is_valid_data(self, data):
        validation_message = {"is_valid": True}

        if not isinstance(data, numbers.Number):
            validation_message["is_valid"] = False
        validation_message["error"] = "Data fed to Rescale is not Number " \
                                      "Object. Please make sure if the data " \
                                      "being fed to the Rescale Processor is " \
                                      "a number. E.g., int, float, decimal, " \
                                      "etc."
        return validation_message

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        return rescale([data], self.new_range, self.original_range)[0]



