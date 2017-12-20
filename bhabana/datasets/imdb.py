import os

import bhabana.utils as utils
import bhabana.utils.data_utils as du

from bhabana.processing import *


class IMDB:
    name = "imdb"
    lang = "en"
    size = "medium"
    dataset_root_dir = os.path.join(utils.DATASET_DIR, name)
    supported_vocabs = ["word", "char", "semhash", "pos", "ent", "dep"]
    vocab_paths = {
        key: os.path.join(dataset_root_dir, "{}_vocab.txt".format(key))
        for key in supported_vocabs
    }
    splits = {"train": os.path.join(dataset_root_dir, "train"),
               "test": os.path.join(dataset_root_dir, "test"),
               "validation": os.path.join(dataset_root_dir, "validation")}
    classes_path = os.path.join(dataset_root_dir, "classes.txt")

    def __init__(self,mode="regression", use_spacy_vocab=True,
                 use_spacy_vectors=False, spacy_model_name=None, aux=[],
                 pad_sequences=True, number_of_workers=1, torch_tensors=True,
                 cuda=True, normalize=True):
        du.maybe_download(self.name, type="dataset")
        self.spacy_model_name = spacy_model_name
        self.use_spacy_vocab = use_spacy_vocab
        self.use_spacy_vectors = use_spacy_vectors
        self.aux = self._set_aux(aux)
        self._set_mode(mode)
        self.pad_sequences = pad_sequences
        self.number_of_workers = number_of_workers
        self.torch_tensors = torch_tensors
        self.cuda = cuda
        self.normalize = normalize
        if self.use_spacy_vocab:
            self._load_spacy_vocabulary()
        self._load_vocabularies()
        self._load_classes()
        self.w2v = self._load_word_vectors() if use_spacy_vectors else None

    def valid_aux(self, aux):
        ret = True
        for a in aux:
            if a not in self.supported_vocabs:
                ret = False
                break
        return ret

    def _set_aux(self, aux):
        if not self.valid_aux(aux):
            raise ValueError("aux contains invalid values. The following are "
                             "the values you have passed: \n {}. Please "
                             "provide valid auxiliaries in aux."
                             " The following are valid "
                             "aux values: {}".format(", ".join(aux),
                             ", ".join([k for k in self.supported_vocabs
                                             if k != "word"])))
        else:
            self.aux = list(set(aux + ["word"])) if "word" not in aux\
                                                 else list(set(aux))

    def _set_mode(self, mode):
        if mode == "regression" or mode == "classification":
            self.mode = mode.lower()
        else:
            raise ValueError("The mode that you have set is {}. It is an "
                 "invalid mode for the IMDB Dataset. This dataset can be loaded"
                 " in only two modes. (1) Classiification and (2) Regression.")

    def _load_word_vectors(self):
        return du.preload_w2v(self.vocab["word"][0], lang=self.lang,
                              model=self.spacy_model_name)

    def _load_aux_vocabularies(self):
        pass

    def _load_spacy_vocabulary(self):
        self.vocab["word"] = os.path.join(self.dataset_root_dir,
                                                "spacy_word_vocab.txt")
        du.write_spacy_vocab(self.vocab["word"], self.lang,
                             model_name=self.spacy_model_name)

    def _load_vocabularies(self):
        self.vocab = {
                key: du.load_vocabulary(self.vocab_paths[key])
                for key in self.vocab_paths if key in self.aux
            }
        self.vocab_sizes = {
            key: len(self.vocab[key])
            for key in self.vocab if key in self.aux
        }

    def _load_classes(self):
        self.c2i, self.i2c = du.load_classes(self.classes_path)
        self.n_classes = len(self.c2i)

    def _get_pipeline(self, type):
        if type in self.supported_vocabs:
            pipeline = [Tokenizer(lang=self.lang,
                                   spacy_model_name=self.spacy_model_name,
                                   mode=type),
                        Seq2Id(self.vocab[type][0])]
        elif type == "regression":
            pipeline = []
        elif type == "classification":
            pipeline = [Class2Id(c2i=self.c2i),
                        OneHot(n_classes=self.n_classes)]
        else:
            raise NotImplementedError("{} pipeline has not been implemented "
                                      "yet or it is an invalid pipeline type "
                                      "for the IMDB Dataset")
        return pipeline

    def _get_text_processing_pipelines(self):
        text_processing_pipeline = []
        for aux in self.aux:
            text_processing_pipeline.append(DataProcessingPipeline(
                    pipeline=self._get_pipeline(aux),
                    name= "text" if aux == "word" else aux,
                    add_to_output=True
            ))
        return text_processing_pipeline

    def load_fields(self):
        self.fields = [
            {
                "key": "text", "dtype": str,
                "processors":self._get_text_processing_pipelines()
            },
            {
                "key": "sentiment", "dtype": str,
                "processors": DataProcessingPipeline(self._get_pipeline(
                        self.mode), name="label", add_to_output=True)
            }
        ]