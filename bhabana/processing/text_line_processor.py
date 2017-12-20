import re
import json

from bhabana.processing import Processor


class TextLineProcessor(Processor):
    def __init__(self, fields, separator="\t"):
        self.fields = fields
        self.separator = separator

    def is_valid_data(self, data):
        validation_message = {"is_valid": True}
        if type(data) != str:
            raise ValueError("input data must be a string")
        find = re.findall(r"".format(self.separator), data)
        if len(find) == len(self.fields) - 1:
            raise ValueError('input data does not have enough columns to '
                             'match the number of fields. Please verify '
                             'your data or check whether you have set the '
                             'fields right or not.')
        return validation_message

    def __call__(self, data):
        self.process(data)

    def process(self, data):
        cols = data.split(self.separator)
        response = []
        for field, col in zip(self.fields, cols):
            response.append(field["dtype"](col))
        return response


