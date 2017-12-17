import logging

import numpy as np
import torch as th
import torch.nn as nn

from nose.tools import *
from torch.autograd import Variable
from bhabana.utils import generic_utils as gu

logger = logging.getLogger(__name__)


class TestGenericUtils():

    @raises(ValueError)
    def test_invalid_activation(self):
        gu.get_activation_by_name("blah")

    def test_valid_activation_none(self):
        act = gu.get_activation_by_name(None)
        assert_equals(act, None)

    def test_valid_activation_sigmoid(self):
        act = gu.get_activation_by_name("sigmoid")
        assert_equals(act, nn.Sigmoid)

    def test_valid_to_categorical(self):
        lbls = [1, 2, 3]
        one_hot_enc_vecs = gu.to_categorical(lbls, 4)
        gt_1 = np.array([0, 1, 0, 0])
        gt_2 = np.array([0, 0, 1, 0])
        gt_3 = np.array([0, 0, 0, 1])
        assert_equals(one_hot_enc_vecs.shape[0], 3)
        assert_equals(one_hot_enc_vecs.shape[1], 4)
        assert_true(np.array_equal(one_hot_enc_vecs[0], gt_1))
        assert_true(np.array_equal(one_hot_enc_vecs[1], gt_2))
        assert_true(np.array_equal(one_hot_enc_vecs[2], gt_3))

        lbl = 2
        ohev = gu.to_categorical(lbl, 4)
        assert_equals(ohev.shape[0], 4)
        assert_true(np.array_equal(ohev, gt_2))
