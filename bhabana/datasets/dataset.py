import os
import torch

import bhabana.utils as utils
import bhabana.utils.data_utils as du

from bhabana.processing import *
from abc import abstractmethod
from bhabana.datasets import TextDataset


class Dataset():

    def __init__(self, name, lang="en", size="medium", mode="regression",
                 use_spacy_vocab=True, load_spacy_vectors=False,
                 spacy_model_name=None, aux=[], cuda=True, rescale=None):
        self.name = name
        self.lang = lang
        self.size = size
        self._boilerplate()
        du.maybe_download(self.name, type="dataset")
        self.spacy_model_name = spacy_model_name
        self.use_spacy_vocab = use_spacy_vocab
        self.load_spacy_vectors = load_spacy_vectors
        self.aux = self._set_aux(aux)
        self._set_mode(mode)
        self.cuda = True if cuda and torch.cuda.is_available() else False
        self.rescale = rescale
        if self.use_spacy_vocab:
            self._load_spacy_vocabulary()
        self._load_aux_vocabularies()
        self._load_w2is()
        self._load_classes()
        self.w2v = self._load_word_vectors() if load_spacy_vectors else None
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.cuda else torch.ByteTensor
        self.train = None
        self.test = None
        self.validation = None

    def _boilerplate(self):
        self.dataset_root_dir = os.path.join(utils.DATASET_DIR,self. name)
        self.supported_vocabs = ["word", "char", "semhash", "pos", "ent", "dep"]
        self.vocab_paths = {
            key: os.path.join(self.dataset_root_dir, "{}_vocab.txt".format(key))
            for key in self.supported_vocabs
        }
        self.splits = {"train": os.path.join(self.dataset_root_dir, "train"),
                  "test": os.path.join(self.dataset_root_dir, "test"),
                  "validation": os.path.join(self.dataset_root_dir, "validation")}
        self.classes_path = os.path.join(self.dataset_root_dir, "classes.txt")

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
                                                     ", ".join([k for k in
                                                                self.supported_vocabs
                                                                if
                                                                k != "word"])))
        else:
            self.aux = list(set(aux + ["word"])) if "word" not in aux \
                else list(set(aux))

    def _set_mode(self, mode):
        if mode == "regression" or mode == "classification":
            self.mode = mode.lower()
        else:
            raise ValueError("The mode that you have set is {}. It is an "
                             "invalid mode for the IMDB Dataset. This dataset "
                             "can be loaded"
                             " in only two modes. (1) Classiification and (2) "
                             "Regression.")

    def _load_word_vectors(self):
        return du.preload_w2v(self.vocab["word"][0], lang=self.lang,
                              model=self.spacy_model_name)

    def _load_aux_vocabularies(self):
        for aux in self.aux:
            du.write_spacy_aux_vocab(self.vocab["aux"], self.lang, aux)

    def _load_spacy_vocabulary(self):
        self.vocab["word"] = os.path.join(self.dataset_root_dir,
                                          "spacy_word_vocab.txt")
        du.write_spacy_vocab(self.vocab["word"], self.lang,
                             model_name=self.spacy_model_name)

    def _load_w2is(self):
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

    def initialize_splits(self, fields, line_processor):
        if os.path.exists(self.splits["train"]):
            self.train = TextDataset(self.splits["train"], fields,
                                     line_processor)
        if os.path.exists(self.splits["test"]):
            self.test = TextDataset(self.splits["test"], fields,
                                    line_processor)
        if os.path.exists(self.splits["train"]):
            self.train = TextDataset(self.splits["validation"],
                                     fields, line_processor)

    def get_batch(self, torch_tensor=True, pad=True, n_workers=1,
                  batch_size=32):
        pass

    def validate_fields(self):
        # TODO: Implement and validate fields
        pass

    @abstractmethod
    def load_fields(self):
        pass