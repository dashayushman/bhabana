import json

from bhabana.processing import Processor


class JSONLineProcessor(Processor):
    def __init__(self, fields):
        self.fields = fields

    def is_valid_data(self, data):
        validation_message = {"is_valid": True}
        try:
            j = json.loads(data, encoding="utf-8")
            for i_f, field in enumerate(self.fields):
                if field["key"] not in j:
                    raise ValueError('{} was set as the key in field {}, '
                                     'but it was not found in the data. '
                                     'Please verify if the data is correct or'
                                     ' if the field is missing a key'.format(
                                        field["key"], i_f))
                if field["key"] is None:
                    raise ValueError('Field {} with key {} has value None. '
                                     'Please verify if your data is in the '
                                     'right format'.format(i_f, field["key"]))
        except Exception as e:
            validation_message["is_valid"] = False
            validation_message["error"] = "Data must be a valid JSON String. " \
                                          "Error Message: \n" \
                                          "{}".format(str(e))
        return validation_message

    def __call__(self, data):
        self.process(data)

    def process(self, data):
        j = json.loads(data, encoding="utf-8")
        response = []
        for field in self.fields:
            response.append(field["dtype"](j[field["key"]]))
        return response


