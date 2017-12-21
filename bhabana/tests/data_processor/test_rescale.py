from nose.tools import *
from bhabana.processing import Rescale


class TestRescale:
    original_lb = 10
    original_ub = 20

    new_lb = 0
    new_ub = 1

    def setUp(self):
        self.rescale = Rescale(self.original_lb, self.original_ub,
                               self.new_lb, self.new_ub)

    def test_rescale(self):
        val = 12
        rescaled_val = self.rescale.process(val)
        assert_true(rescaled_val >= self.new_lb and rescaled_val <= self.new_ub)
        assert_equals(rescaled_val, .2)
