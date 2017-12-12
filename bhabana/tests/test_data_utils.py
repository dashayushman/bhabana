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
    invalid_tar_file_path = os.path.join(temp_dir, invalid_tar_file)

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




