import torch as th


from nose.tools import *
from torch.autograd import Variable
from bhabana.models import Regressor


class TestRegressionModel():
    input_size = 300
    activation = "sigmoid"
    n_samples = 20
    inputs = Variable(th.randn(n_samples, input_size))

    def setUp(self):
        self.reg = Regressor(input_size=self.input_size,
                             activation=self.activation)

    def tearDown(self):
        self.reg = None

    @raises(Exception)
    def test_invalid_constructor_input_size_neg(self):
        self.reg = Regressor(input_size=-234,
                             activation=self.activation)

    @raises(Exception)
    def test_invalid_constructor_input_size_0(self):
        self.reg = Regressor(input_size=0,
                             activation=self.activation)

    @raises(ValueError)
    def test_invalid_constructor_activation(self):
        self.reg = Regressor(input_size=self.input_size,
                             activation="blah")

    def test_valid_forward(self):
        data = {"inputs": self.inputs}
        response = self.reg(data)
        assert_not_equals(response, None)
        for ret in self.reg.provides:
            assert_true(ret in response)
        out_shape = list(response["out"].size())
        assert_equals(out_shape[0], self.n_samples)
        assert_equals(out_shape[1], 1)

    def test_valid_forward_linear_activation(self):
        self.reg = Regressor(input_size=self.input_size,
                             activation=None)
        data = {"inputs": self.inputs}
        response = self.reg(data)
        assert_not_equals(response, None)
        for ret in self.reg.provides:
            assert_true(ret in response)
        out_shape = list(response["out"].size())
        assert_equals(out_shape[0], self.n_samples)
        assert_equals(out_shape[1], 1)
        assert_equals(self.reg.activation, None)
