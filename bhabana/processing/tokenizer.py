from bhabana.utils.data_utils import tokenize
from bhabana.utils.data_utils import sentence_tokenize
from bhabana.utils.data_utils import semhash_tokenize
from bhabana.utils.data_utils import char_tokenize
from bhabana.utils.data_utils import pos_tokenize
from bhabana.utils.data_utils import dep_tokenize
from bhabana.utils.data_utils import ent_tokenize

from bhabana.processing import Processor


class Tokenizer(Processor):

    def __init__(self, lang="en", tokenizer="spacy",
                 spacy_model_name=None, mode="word", process_batch=False):
        self.lang = lang
        self.tokenizer = tokenizer
        self.spacy_model_name = spacy_model_name
        self.mode = mode
        self.batch = process_batch

    def is_valid_data(self, data):
        validation_message = {"is_valid": True}
        if type(data) != str:
            validation_message["is_valid"] = False
            validation_message["error"] = "Data fed to Tokenizer is not a " \
                                          "String. Ideally the data being " \
                                          "fed to a Tokenizer in a " \
                                          "DataProcessingPipeline must " \
                                          "must be a String object"
        return validation_message

    def __call__(self, data):
        return self.process(data)

    def _process_batch(self, batch_data):
        if self.mode == "word":
            tokens = [tokenize(data, tokenizer=self.tokenizer,
                               lang=self.lang) for data in batch_data]
        elif self.mode == "char":
            tokens = [char_tokenize(data) for data in batch_data]
        elif self.mode == "semhash":
            tokens = [semhash_tokenize(data, tokenizer=self.tokenizer,
                                      lang=self.lang) for data in batch_data]
        elif self.mode == "sentence":
            tokens = [sentence_tokenize(data, lang=self.lang) for data in batch_data]
        else:
            raise NotImplementedError("Tokenizer for mode '{}' has not been "
                                      "implemented yet, or it is an invalid "
                                      "mode. Please check if you have made a "
                                      "typo.")
        return tokens

    def process(self, data):
        if self.batch:
            tokens = self._process_batch(batch_data=data)
        else:
            if self.mode == "word":
                tokens = tokenize(data, tokenizer=self.tokenizer, lang=self.lang)
            elif self.mode == "char":
                tokens = char_tokenize(data)
            elif self.mode == "semhash":
                tokens = semhash_tokenize(data, tokenizer=self.tokenizer,
                                          lang=self.lang)
            elif self.mode == "sentence":
                tokens = sentence_tokenize(data, lang=self.lang)
            elif self.mode == "pos":
                tokens = pos_tokenize(data, lang=self.lang)
            elif self.mode == "ent":
                tokens = ent_tokenize(data, lang=self.lang)
            elif self.mode == "dep":
                tokens = dep_tokenize(data, lang=self.lang)
            else:
                raise NotImplementedError("Tokenizer for mode '{}' has not been "
                                          "implemented yet, or it is an invalid "
                                          "mode. Please check if you have made a "
                                          "typo.")
        return tokens
