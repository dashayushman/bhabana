from bhabana.utils.data_utils import seq2id
from bhabana.utils.data_utils import semhashseq2id
from bhabana.utils.data_utils import sentence2id

from bhabana.processing import Processor


class Seq2Id(Processor):
    def __init__(self, w2i, seq_begin=False, seq_end=False, mode="word"):
        self.w2i = w2i
        self.seq_begin = seq_begin
        self.seq_end = seq_end
        self.mode = mode

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
        if self.mode in ["word", "char", "dep", "pos", "ent"]:
            idseq = seq2id([data], self.w2i, self.seq_begin, self.seq_end)[0]
        elif self.mode == "semhash":
            idseq = semhashseq2id([data], self.w2i)[0]
        elif self.mode == "sentence":
            idseq = sentence2id([data], self.w2i)[0]
        else:
            raise NotImplementedError("{} mode has not been implemented yet "
                                      "or it is an invalid mode".format(
                    self.mode))
        return idseq