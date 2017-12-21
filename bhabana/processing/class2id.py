from bhabana.processing import Processor


class Class2Id(Processor):
    def __init__(self, c2i):
        self.c2i = c2i

    def is_valid_data(self, data):
        validation_message = {"is_valid": True}
        if type(data) != str:
            validation_message["is_valid"] = False
            validation_message["error"] = "Data fed to Class2Id is not an " \
                      "iterable. Ideally the data being fed to Seq2Id in a " \
                      "DataProcessingPipeline must already be tokenized and" \
                      " it should be an iterable of strings"
        return validation_message

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if data not in self.c2i:
            raise Exception("Class label {} is not in the registered list of "
                        "classes. Please check if it is a valid classs "
                        "label or whether it is missing from the "
                        "classes.txt in the dataset directory.".format(data))
        else:
            return self.c2i[data]


