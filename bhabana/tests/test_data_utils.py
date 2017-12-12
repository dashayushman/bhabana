import os
import logging
import shutil

import bhabana.utils as utils

from nose.tools import *
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




