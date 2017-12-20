import os
import logging
import shutil

import numpy as np
import torch as th
import torch.nn as nn
import bhabana.utils as utils

from nose.tools import *
from torch.autograd import Variable
from bhabana.utils import data_utils as du

logger = logging.getLogger(__name__)


class TestDataUtils():
    temp_dir = os.path.join("/tmp", "bhabana_data")
    valid_file_url = utils.BASE_URL + 'datasets/' + 'test_ds.tar.gz'
    valid_file_name = "test_ds.tar.gz"
    invalid_tar_file = "test.tar.gz"
    untared_file_name = "test_ds"
    untared_path = os.path.join(temp_dir, 'test_ds')
    invalid_tar_file_path = os.path.join(temp_dir, invalid_tar_file)
    valid_tar_file_path = os.path.join(temp_dir, valid_file_name)

    def __init__(self):
        self.download_valid_file_from_url_setup()

    def __del__(self):
        self.download_valid_file_from_url_teardown()

    @classmethod
    def download_valid_file_from_url_setup(self):
        if not os.path.exists(os.path.join(self.temp_dir,
                                           self.valid_file_name)):
            du.download_from_url(self.valid_file_url, self.temp_dir)

            with open(self.invalid_tar_file_path, 'w') as f:
                f.write("This is an invalid tar file")

    @classmethod
    def download_valid_file_from_url_teardown(self):
        tar_path = os.path.join(self.temp_dir, self.valid_file_name)
        untared_path = os.path.join(utils.DATASET_DIR, 'test_ds')
        if os.path.exists(tar_path): os.remove(tar_path)
        if os.path.exists(untared_path): shutil.rmtree(untared_path)
        if os.path.exists(self.temp_dir): shutil.rmtree(self.temp_dir)

    @classmethod
    def validate_sequence(self, original_sequence, padded_seq, max_len,
                          gt_dtype, gt_pad_value, pad_index):
        assert_equal(len(padded_seq[0]), max_len)
        assert_equal(len(padded_seq[1]), max_len)
        if max_len > len(original_sequence[0]):
            assert_equal(padded_seq[0][pad_index], gt_pad_value)
        if max_len > len(original_sequence[0]):
            assert_equal(padded_seq[1][pad_index], gt_pad_value)
        assert_equal(type(padded_seq[0][pad_index]).__name__, gt_dtype)
        assert_equal(type(padded_seq[1][pad_index]).__name__, gt_dtype)

    @classmethod
    def validate_pad_sequence(self, original_sequence, padded_seq, max_len,
                              gt_pad_value, pad_index=-1, raw=False):
        assert_equal(len(padded_seq[0]), max_len)
        if max_len > len(original_sequence[0]):
            assert_equal(padded_seq[0][pad_index], gt_pad_value)
        if raw and max_len > len(original_sequence[0]):
            assert_equal(padded_seq[0][pad_index], gt_pad_value)

    @raises(Exception)
    def test_invalid_url(self):
        assert_false(du.url_exists("dsafcsef"))

    # TODO: add a new assert for valid_model_url
    def test_valid_url(self):
        valid_base_url = utils.BASE_URL
        valid_datasets_url = utils.BASE_URL + 'datasets'
        valid_models_url = utils.BASE_URL + 'models'
        valid_dataset_url = utils.BASE_URL + 'datasets/' + 'test_ds.tar.gz'
        assert_true(du.url_exists(valid_base_url))
        assert_true(du.url_exists(valid_datasets_url))
        assert_true(du.url_exists(valid_models_url))
        assert_true(du.url_exists(valid_dataset_url))

    def test_forced_user_download_response(self):
        assert_true(du.user_wants_to_download("xyz", force=True))

    @raises(Exception)
    def test_download_from_invalid_url(self):
        du.download_from_url("dkjahcb", "./")

    def test_download_from_valid_url(self):
        assert_true(os.path.exists(self.temp_dir))
        assert_true(os.path.exists(os.path.join(self.temp_dir,
                                                self.valid_file_name)))
    @raises(FileNotFoundError)
    def test_extract_non_existent_file(self):
        du.extract_tar_gz("sdvsd")

    @raises(ValueError)
    def test_extract_invalid_tar_file(self):
        du.extract_tar_gz(self.invalid_tar_file_path)

    @raises(ValueError)
    def test_extract_invalid_tar_file(self):
        du.extract_tar_gz(self.invalid_tar_file_path)

    def test_extract_valid_tar_invalid_output_dir(self):
        output_dir = os.path.join(self.temp_dir, 'test', 'test')
        du.extract_tar_gz(self.valid_tar_file_path, output_dir)
        untared_file_path = os.path.join(output_dir, self.untared_file_name)
        assert_true(os.path.exists(untared_file_path))
        if os.path.exists(untared_file_path): shutil.rmtree(untared_file_path)

    def test_extract_valid_tar(self):
        du.extract_tar_gz(self.valid_tar_file_path, self.temp_dir)
        assert_true(os.path.exists(self.untared_path))

    @raises(Exception)
    def test_load_invalid_spacy_lang(self):
        du.get_spacy("blah")

    @raises(Exception)
    def test_load_invalid_spacy_model_and_lang(self):
        du.get_spacy("blah", "blah")

    @raises(Exception)
    def test_load_invalid_spacy_model_valid_lang(self):
        du.get_spacy("en", "blah")

    def test_load_valid_spacy_model_valid_lang(self):
        spacy_obj = du.get_spacy("de", "de_core_news_sm")
        assert_not_equal(spacy_obj, None)
        key = "de_de_core_news_sm"
        assert_not_equal(du.spacy_nlp_collection[key], None)
        assert_true(key in du.spacy_nlp_collection)
        del du.spacy_nlp_collection[key]

    @raises(ValueError)
    def test_load_spacy_not_corresponding_lang_model(self):
        du.get_spacy("en", "de_core_news_sm")

    @raises(Exception)
    def test_invalid_pad_int_sequence_batch(self):
        sequence = [0, 1, 3, 5]
        #sequence = ["hello", "hello", "hello"]
        du.pad_int_sequences(sequences=sequence)

    @raises(Exception)
    def test_invalid_trunc_pad_int_sequence_batch(self):
        sequence = [0, 1, 3, 5]
        du.pad_int_sequences(sequences=[sequence], truncating="blah")

    @raises(Exception)
    def test_invalid_padding_pad_int_sequence_batch(self):
        sequence = [0, 1, 3, 5]
        du.pad_int_sequences(sequences=[sequence], padding="blah")

    @raises(Exception)
    def test_invalid_padlen_zero_pad_int_sequence_batch(self):
        sequence = [0, 1, 3, 5]
        du.pad_int_sequences(sequences=[sequence], padding="post",
                             maxlen=-1)

    def test_zero_len_pad_int_sequence_batch(self):
        sequence = [0, 1, 3, 5]
        padded_seq = du.pad_int_sequences(sequences=[sequence],
                                          padding="post", maxlen=0)
        assert_equal(len(padded_seq[0]), 0)


    def test_valid_pad_int_sequence_batch_default_params(self):
        sequence = [[0, 1, 3, 5],
                    [9, 4, 5, 23, 43]]
        # sequence = ["hello", "hello", "hello"]
        padded_sequence = du.pad_int_sequences(sequences=sequence,
                                               maxlen=10, dtype="int32", padding="post",
                                               truncating="post", value= 0)
        assert_equal(len(padded_sequence[0]), 10)
        assert_equal(len(padded_sequence[1]), 10)
        assert_equal(padded_sequence[0][-1], 0)
        assert_equal(padded_sequence[1][-1], 0)
        assert_equal(type(padded_sequence[0][-1]).__name__, "int32")
        assert_equal(type(padded_sequence[1][-1]).__name__, "int32")

    def test_valid_pad_int_sequence_batch(self):
        sequence = [[0, 1, 3, 5],
                    [9, 4, 5, 23, 43]]
        # sequence = ["hello", "hello", "hello"]
        padded_sequence = du.pad_int_sequences(sequences=sequence,
                                               maxlen=10, dtype="int32", padding="post",
                                               truncating="post", value=0)
        self.validate_sequence(sequence, padded_sequence, max_len=10,
                               gt_dtype="int32", gt_pad_value=0, pad_index=-1)

        padded_sequence = du.pad_int_sequences(sequences=sequence,
                                               maxlen=2, dtype="float64", padding="post",
                                               truncating="post", value=0)
        self.validate_sequence(sequence, padded_sequence, max_len=2,
                               gt_dtype="float64", gt_pad_value=0, pad_index=-1)

        padded_sequence = du.pad_int_sequences(sequences=sequence,
                                               maxlen=2, dtype="float64", padding="post",
                                               truncating="pre", value=0)
        self.validate_sequence(sequence, padded_sequence, max_len=2,
                               gt_dtype="float64", gt_pad_value=0, pad_index=-1)

        padded_sequence = du.pad_int_sequences(sequences=sequence,
                                               maxlen=2, dtype="float64", padding="pre",
                                               truncating="pre", value=4)
        self.validate_sequence(sequence, padded_sequence, max_len=2,
                               gt_dtype="float64", gt_pad_value=4, pad_index=0)

        padded_sequence = du.pad_int_sequences(sequences=sequence,
                                               maxlen=10, dtype="float64", padding="pre",
                                               truncating="post", value=4)
        self.validate_sequence(sequence, padded_sequence, max_len=10,
                               gt_dtype="float64", gt_pad_value=4, pad_index=0)

    @raises(Exception)
    def test_invalid_pad_sequences(self):
        du.pad_sequences([1, 2, 3, 4, "hello"], padlen=-1)

    @raises(Exception)
    def test_invalid_sequence_pad_sequences(self):
        du.pad_sequences([1, 2, 3, 4, "hello"], padlen=20, raw=False)

    def test_valid_pad_sequences(self):
        int_sequence = [[1, 2, 3, 4]]
        str_sequence = [[1, 2, 3, 4, "hello"]]
        padded_sequence_1 = du.pad_sequences(int_sequence, padlen=20,
                                             padvalue=486, raw=False)
        padded_sequence_2 = du.pad_sequences(str_sequence, padlen=2,
                                             padvalue="asd", raw=True)
        padded_sequence_3 = du.pad_sequences(str_sequence, padlen=100,
                                             padvalue=1, raw=True)
        self.validate_pad_sequence(int_sequence, padded_sequence_1,
                                   max_len=20, gt_pad_value=486)
        self.validate_pad_sequence(str_sequence, padded_sequence_2,
                                   max_len=2, gt_pad_value="PAD")
        self.validate_pad_sequence(str_sequence, padded_sequence_3,
                                   max_len=100, gt_pad_value="PAD")

    @raises(Exception)
    def test_valid_pad_conv_1d(self):
        input_seq = Variable(th.randn(10, 20, 30))
        du.pad_1dconv_input(input_seq, mode="valid")

    @raises(Exception)
    def test_full_pad_conv_1d_stride_1(self):
        input_seq = Variable(th.randn(10, 20, 30))
        du.pad_1dconv_input(input_seq, mode="full")

    @raises(Exception)
    def test_wrong_pad_conv_1d_stride_1(self):
        input_seq = Variable(th.randn(10, 20, 30))
        du.pad_1dconv_input(input_seq, mode="blah")

    @raises(Exception)
    def test_wrong_input_pad_conv_1d_stride_1(self):
        input_seq = Variable(th.randn(10, 20, 30, 34))
        du.pad_1dconv_input(input_seq, kernel_size=4, mode="same")

    def test_1dconv_pad_same_conv_stride_1(self):
        # BATCH X TIME_STEPS X IN_CHANNELS
        input_seq = Variable(th.randn(10, 20, 30))
        padded_input_seq = du.pad_1dconv_input(input_seq, kernel_size=4,
                                               mode="same")
        # BATCH X IN_CHANNELS X TIME_STEPS
        padded_input_seq = padded_input_seq.permute(0, 2, 1)
        conv = nn.Conv1d(30, 100, 4)
        conv_output = conv(padded_input_seq)
        conv_output_shape = list(conv_output.size())
        assert_equals(conv_output_shape[0], 10)
        assert_equals(conv_output_shape[1], 100)
        assert_equals(conv_output_shape[2], 20)

        padded_input_seq = du.pad_1dconv_input(input_seq, kernel_size=5,
                                               mode="same")
        # BATCH X IN_CHANNELS X TIME_STEPS
        padded_input_seq = padded_input_seq.permute(0, 2, 1)
        conv = nn.Conv1d(30, 100, 5)
        conv_output = conv(padded_input_seq)
        conv_output_shape = list(conv_output.size())
        assert_equals(conv_output_shape[0], 10)
        assert_equals(conv_output_shape[1], 100)
        assert_equals(conv_output_shape[2], 20)

    def test_1dconv_pad_full_conv_stride_1(self):
        # BATCH X TIME_STEPS X IN_CHANNELS
        input_seq = Variable(th.randn(10, 20, 30))
        padded_input_seq = du.pad_1dconv_input(input_seq, kernel_size=5,
                                               mode="full")
        padded_seq_shape = list(padded_input_seq.size())
        # BATCH X IN_CHANNELS X TIME_STEPS
        padded_input_seq = padded_input_seq.permute(0, 2, 1)
        conv = nn.Conv1d(30, 100, 5)
        conv_output = conv(padded_input_seq)
        conv_output_shape = list(conv_output.size())
        assert_equals(conv_output_shape[0], 10)
        assert_equals(conv_output_shape[1], 100)
        assert_equals(conv_output_shape[2], padded_seq_shape[1] - 5 + 1)

    def test_write_spacy_vocab(self):
        path = '/tmp/spacy_vocab.txt'
        du.write_spacy_vocab(path, lang="de")
        assert_true(os.path.exists(path))

    def test_load_w2v(self):
        w2i = {"blah": 0, "heansszclk!!": 1}
        w2v = du.preload_w2v(w2i, lang="de", model=None)
        size = du.get_spacy_vector_size("de", None)
        assert_equals(w2v.shape[0], 2)
        assert_true(w2v.shape[1], size)

    def test_get_spacy_vector_size(self):
        size = du.get_spacy_vector_size("de", None)
        assert_not_equals(size, None)
