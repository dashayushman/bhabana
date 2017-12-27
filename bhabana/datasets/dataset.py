import os
import torch

import numpy as np
import bhabana.utils as utils
import bhabana.utils.data_utils as du

from torch.autograd import Variable
from bhabana.datasets import TextDataset
from torch.utils.data.dataloader import DataLoader


class Dataset():

    def __init__(self, name, lang="en", size="medium", mode="regression",
                 use_spacy_vocab=True, load_spacy_vectors=False,
                 spacy_model_name=None, aux=[], cuda=True, rescale=None):
        self.name = name
        self.lang = lang
        self.size = size
        du.maybe_download(self.name, type="dataset")
        self.spacy_model_name = spacy_model_name
        self.use_spacy_vocab = use_spacy_vocab
        self.load_spacy_vectors = load_spacy_vectors
        self._boilerplate()
        self._set_aux(aux)
        self._set_mode(mode)
        self.cuda = True if cuda and torch.cuda.is_available() else False
        self.rescale = rescale
        if self.use_spacy_vocab:
            self._load_spacy_vocabulary()
        self._load_aux_vocabularies()
        self._load_w2is()
        self._load_classes()
        self.w2v = self._load_word_vectors() if load_spacy_vectors else None
        self.train = None
        self.test = None
        self.validation = None
        self.dataloaders = {"train": None, "test": None, "validation": None}

    def _boilerplate(self):
        self.dataset_root_dir = os.path.join(utils.DATASET_DIR,self. name)
        self.supported_vocabs = ["word", "char", "semhash", "pos", "ent", "dep"]
        self.supported_aux = self.supported_vocabs + ["sentence"]
        self.vocab_paths = {
            key: os.path.join(self.dataset_root_dir, "{}_vocab.txt".format(key))
            for key in self.supported_vocabs
        }
        self.splits = {"train": os.path.join(self.dataset_root_dir, "train"),
                  "test": os.path.join(self.dataset_root_dir, "test"),
                  "validation": os.path.join(self.dataset_root_dir, "validation")}
        self.classes_path = os.path.join(self.dataset_root_dir, "classes.txt")
        self._set_w2v_path()

    def _set_w2v_path(self):
        if self.load_spacy_vectors and not self.use_spacy_vocab:
            self.w2v_path = os.path.join(self.dataset_root_dir, "spacy_w2v.npy")
        elif self.load_spacy_vectors and self.use_spacy_vocab:
            self.w2v_path = os.path.join(self.dataset_root_dir,
                                         "spacy_w2v_spacy_vocab.npy")
        else:
            self.w2v_path = os.path.join(self.dataset_root_dir, "w2v.npy")

    def valid_aux(self, aux):
        ret = True
        for a in aux:
            if a not in self.supported_aux:
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
            aux = [a.lower() for a in aux]
            self.aux = list(set(aux + ["word"])) if "word" not in aux \
                else list(set(aux))

    def _set_mode(self, mode):
        if mode in ["regression", "classification"]:
            self.mode = mode.lower()
        else:
            raise ValueError("The mode that you have set is {}. It is an "
                             "invalid mode for the IMDB Dataset. This dataset "
                             "can be loaded"
                             " in only two modes. (1) Classiification and (2) "
                             "Regression.")

    def _load_word_vectors(self):
        if os.path.exists(self.w2v_path):
            w2v = np.load(self.w2v_path)
        else:
            w2v = du.preload_w2v(self.vocab["word"][0], lang=self.lang,
                                  model=self.spacy_model_name)
            np.save(self.w2v_path, w2v)
        return w2v

    def _load_aux_vocabularies(self):
        for aux in self.aux:
            if aux in self.supported_vocabs and aux not in ["semhash",
                                                            "word", "char"]:
                du.write_spacy_aux_vocab(self.vocab_paths[aux], self.lang, aux)

    def _load_spacy_vocabulary(self):
        self.vocab_paths["word"] = os.path.join(self.dataset_root_dir,
                                          "spacy_word_vocab.txt")
        du.write_spacy_vocab(self.vocab_paths["word"], self.lang,
                                         model_name=self.spacy_model_name)

    def _load_w2is(self):
        self.vocab = {
            key: du.load_vocabulary(self.vocab_paths[key])
            for key in self.vocab_paths if key in self.aux
        }
        self.vocab["sentence"] = self.vocab["word"]
        self.vocab_sizes = {
            key: len(self.vocab[key][0])
            for key in self.vocab if key in self.aux
        }

    def _load_classes(self):
        self.c2i, self.i2c = du.load_classes(self.classes_path)
        self.n_classes = len(self.c2i)

    def set_provides(self, fields):
        self.provides = []
        for field in fields:
            if hasattr(field["processors"], "__iter__"):
                for processor in field["processors"]:
                    self.provides.append((processor.name, field["type"]))
                    if processor.add_length:
                        self.provides.append(
                                ("{}_length".format(processor.name),
                                 field["type"]))
            else:
                self.provides.append((field["processors"].name,
                                     field["type"]))
                if field["processors"].add_length:
                    self.provides.append(
                            ("{}_length".format(field["processors"].name),
                             field["type"]))

    def set_fields(self, fields):
        # TODO: vaidate fields
        self.fields = fields

    def initialize_splits(self, line_processor, fields=None):
        if fields is not None:
            self.set_provides(fields)
            self.set_fields(fields)
        if os.path.exists(self.splits["train"]):
            self.train = TextDataset(self.splits["train"], self.fields,
                                     line_processor)
        if os.path.exists(self.splits["test"]):
            self.test = TextDataset(self.splits["test"], self.fields,
                                    line_processor)
        if os.path.exists(self.splits["train"]):
            self.validation = TextDataset(self.splits["validation"],
                                          self.fields, line_processor)

    def _get_data_loader_for_split(self, split="train", batch_size=32,
                                    shuffle=False, num_workers=1):
        def collate_batch(batch):
            collated_data = {}
            for key in batch[0]:
                collated_data[key] = []
                for instance in batch:
                        collated_data[key].append(instance[key])
            return collated_data

        if split == "train" and self.train is not None:
            dataloader = DataLoader(self.train, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers,
                                    collate_fn=collate_batch,
                                    drop_last=True)
        elif split == "test" and self.test is not None:
            dataloader = DataLoader(self.test, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers,
                                    collate_fn=collate_batch,
                                    drop_last=True)
        elif split == "validation" and self.validation is not None:
            dataloader = DataLoader(self.validation, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers,
                                    collate_fn=collate_batch,
                                    drop_last=True)
        else:
            raise Exception("This dataset does not have a {} split".format(split))
        return dataloader

    def _get_dataloader(self, split, num_workers=1, shuffle=True,
                        batch_size=32):
        if split in self.dataloaders:
            if self.dataloaders[split] is None:
                self.dataloaders[split] = self._get_data_loader_for_split(split=split,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     num_workers=num_workers)
                return self.dataloaders[split]
            else:
                return self.dataloaders[split]
        else:
            raise Exception("Invalid SPlit {}".format(split))


    def get_batch(self, split="train", to_tensor=True, pad=True, num_workers=1,
                  shuffle=True, batch_size=32, seq_max_len=0):
        dataloader = self._get_dataloader(split, num_workers, shuffle,batch_size)
        for i_batch, sample_batched in enumerate(dataloader):
            for key, type in self.provides:
                if "length" not in key and pad and type not in ["label",
                                                                "onehot_label"]:
                    if seq_max_len != 0:
                        max_len = seq_max_len
                    else:
                        max_len = 0
                        for val in sample_batched[key]:
                            max_len = len(val) if len(val) > max_len else max_len
                    sample_batched[key] = du.pad_sequences(
                                    sample_batched[key], padlen=max_len)
                    if to_tensor and type == "sequence":
                        if self.cuda:
                            sample_batched[key] = Variable(torch.LongTensor(
                                                sample_batched[key]).pin_memory().cuda())
                        else:
                            sample_batched[key] = Variable(torch.LongTensor(
                                    sample_batched[key]))
                    else:
                        raise Exception("{} type of fields are not supported "
                                        "or it is an invalid field type. "
                                        "Valid field types are 'sequence' and "
                                                    "'label'".format(type))
                elif (type == "label" or "length" in key) and to_tensor:
                    if self.cuda:
                        sample_batched[key] = Variable(torch.FloatTensor(
                                                sample_batched[key]).pin_memory().cuda(),
                                                       requires_grad=False)
                    else:
                        sample_batched[key] = Variable(torch.FloatTensor(
                                sample_batched[key]), requires_grad=False)
                elif (type == "onehot_label") and to_tensor:
                    if self.cuda:
                        sample_batched[key] = Variable(torch.LongTensor(
                                                sample_batched[key]).pin_memory().cuda(),
                                                       requires_grad=False)
                    else:
                        sample_batched[key] = Variable(torch.LongTensor(
                                sample_batched[key]), requires_grad=False)

            yield i_batch, sample_batched

    def validate_fields(self):
        # TODO: Implement and validate fields
        pass