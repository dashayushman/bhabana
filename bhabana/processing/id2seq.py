from bhabana.utils.data_utils import id2seq
from bhabana.utils.data_utils import id2semhashseq
from bhabana.utils.data_utils import id2charseq

from bhabana.processing import Processor


class Id2Seq(Processor):
    def __init__(self, i2w, mode="word", batch=True):
        self.i2w = i2w
        self.mode = mode
        self.batch = batch

    def is_valid_data(self, data):
        validation_message = {"is_valid": True}
        if not hasattr(data, '__iter__'):
            validation_message["is_valid"] = False
            validation_message["error"] = "Data fed to Seq2Id is not an " \
              "iterable. Ideally the data being fed to Seq2Id in a " \
              "DataProcessingPipeline must already be tokenized and it should" \
              " be an iterable of strings"
        return validation_message

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self.mode in ["word", "dep", "pos", "ent"]:
            idseq = id2seq(data, self.i2w)
        elif self.mode == "semhash":
            idseq = id2semhashseq([data], self.i2w)
        elif self.mode == "char":
            idseq = id2charseq(data, self.i2w)
        else:
            raise NotImplementedError("{} mode has not been implemented yet "
                                      "or it is an invalid mode".format(
                    self.mode))
        return idseq