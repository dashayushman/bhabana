import os
import json
import shutil
import codecs


import numpy as np
import bhabana.utils as utils

from bhabana.datasets import TextDataset
from bhabana.processing import *


class TestTextDataset:
    vocab_size = 20
    test_ds_root_dir = os.path.join(utils.DATASET_DIR, "test_ds")
    json_data_dir = os.path.join(test_ds_root_dir, "json_ds")
    tsv_data_dir = os.path.join(test_ds_root_dir, "tsv_ds")
    custom_data_dir = os.path.join(test_ds_root_dir, "custom_data")
    data_dirs = [json_data_dir, tsv_data_dir, custom_data_dir]
    exts = [".json", ".tsv", ".txt"]
    json_data = json.dumps({
        "text": "Dies ist eine Testreihe über",
        "label": 10, "aux": "some auxiliary value",
        "float": 10.87
    }, ensure_ascii=False)

    tsv_data = "Dies ist eine Testreihe über\t10\tsome auxiliary value\t10.87"
    custom_data = "Dies ist eine Testreihe über##$$##10##$$##some auxiliary value##$$##10.87"
    data_list = [json_data, tsv_data, custom_data]

    def create_test_ds(cls):
        if not os.path.exists(cls.test_ds_root_dir):
            os.makedirs(cls.test_ds_root_dir)
        for data_dir in cls.data_dirs:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

    def create_fake_vocab(self):
        word_vocab_path = os.path.join(self.test_ds_root_dir, "word_vocab.txt")
        char_vocab_path = os.path.join(self.test_ds_root_dir, "char_vocab.txt")
        semhash_vocab_path = os.path.join(self.test_ds_root_dir, "semhash_vocab.txt")
        paths = [word_vocab_path, char_vocab_path, semhash_vocab_path]
        for path in paths:
            with codecs.open(path, "w", "utf-8") as fp:
                buff = ""
                for i in range(self.vocab_size):
                    buff += "{}\n".format(i)
                fp.write(buff)

    def create_fake_w2v(self):
        w2v_path = os.path.join(self.test_ds_root_dir, "w2v.npy")
        w2v = np.array(np.ones((self.vocab_size, 300)))
        np.save(w2v_path, w2v)

    def setUp(self):
        self.create_test_ds()
        for data_dir, ext, data in zip(self.data_dirs, self.exts, self.data_list):
            for i in range(10):
                file_path = os.path.join(data_dir, "{}{}".format(i, ext))
                with codecs.open(file_path, "w", "utf-8") as fp:
                    fp.write(data)
        self.create_fake_vocab()
        self.create_fake_w2v()

    def tearDown(self):
        if os.path.exists(self.test_ds_root_dir):
            shutil.rmtree(self.test_ds_root_dir)


    def test_text_dataset_construct_json(self):
        text_processor = DataProcessingPipeline([])
        fields = [
            {"key": "text", "name": "text", "dtype": str, "processor": None},
            {},
            {}
        ]
        #TextDataset(data_root_dir, fields, line_processor)
