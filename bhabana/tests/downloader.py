import os
import logging
import shutil

import bhabana.utils as utils

from nose.tools import *
from bhabana.utils import data_utils as du

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# TODO: write tests for download_model
class TestDownloader():

    @classmethod
    def assert_dataset_components(self, ds_path):
        assert_true(os.path.exists(os.path.join(ds_path, 'train')))
        assert_true(os.path.exists(os.path.join(ds_path, 'validation')))
        assert_true(os.path.exists(os.path.join(ds_path, 'test')))
        assert_true(os.path.exists(os.path.join(ds_path, 'w2v.npy')))
        assert_true(os.path.exists(os.path.join(ds_path, 'vocab.txt')))
        assert_true(os.path.exists(os.path.join(ds_path,
                                                'metadata.txt')))
        assert_true(os.path.exists(os.path.join(ds_path,
                                                'train', 'train.txt')))
        assert_true(os.path.exists(os.path.join(ds_path,
                                                'validation',
                                                'validation.txt')))
        assert_true(os.path.exists(os.path.join(ds_path,
                                                'test', 'test.txt')))
        assert_true(not os.path.exists(os.path.join(utils.DATASET_DIR,
                                                    'test_ds.tar.gz')))

    def setUp(self):
        pass

    def teardown(self):
        tar_path = os.path.join(utils.DATASET_DIR, 'test_ds.tar.gz')
        untared_path = os.path.join(utils.DATASET_DIR, 'test_ds')
        if os.path.exists(tar_path): os.remove(tar_path)
        if os.path.exists(untared_path): shutil.rmtree(untared_path)

    def test_valid_dataset_download(self):
        logging.info('Test: Download a valid dataset')
        downloaded_path = du.maybe_download('test_ds', type='dataset',
                                           force=True)
        expected_path = os.path.join(utils.DATASET_DIR, 'test_ds')
        assert_equals(downloaded_path, expected_path)
        self.assert_dataset_components(downloaded_path)


    @raises(ValueError)
    def test_invalid_type(self):
        logging.info('Test: Download an invalid type of data. E.g., '
                     '"does_not_exisit"')
        du.maybe_download('test_ds', type='does_not_exisit', force=True)

    @raises(FileNotFoundError)
    def test_invalid_dataset_download(self):
        logging.info('Test: Download an invalid dataset')
        du.maybe_download('xyz', type='dataset', force=True)

    def test_untared_dataset_download(self):
        logging.info('Test: Attempt to download a tar file that already exists')
        url = utils.BASE_URL + 'datasets/test_ds.tar.gz'
        output_dir=utils.DATASET_DIR
        expected_path = os.path.join(utils.DATASET_DIR, 'test_ds')
        du.download_from_url(url, output_dir)
        downloaded_path = du.maybe_download('test_ds', type='dataset', force=True)
        assert_equals(downloaded_path, expected_path)
        self.assert_dataset_components(downloaded_path)
