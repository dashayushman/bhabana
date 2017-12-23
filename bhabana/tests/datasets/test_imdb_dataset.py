from nose.tools import *
from bhabana.datasets import IMDB

class TestIMDBDataset:
    def setUp(self):
        self.imdb = IMDB(mode="regression", use_spacy_vocab=False,
                         load_spacy_vectors=False,
                         aux=["word", "pos"],
                         cuda=True, rescale=(0, 1))
        # , "char", "pos", "dep", "ent", "sentence"
    def test_splits(self):
        assert_not_equals(self.imdb.train, None)
        assert_not_equals(self.imdb.validation, None)
        assert_not_equals(self.imdb.test, None)

    def test_get_batch(self):
        train_batch_gen = self.imdb.get_batch(split="train", to_tensor=True,
                                              pad=True, num_workers=1,
                                              shuffle=True, batch_size=2)
        test_batch_gen = self.imdb.get_batch(split="test", to_tensor=True,
                                              pad=True, num_workers=1,
                                              shuffle=True, batch_size=2)
        validation_batch_gen = self.imdb.get_batch(split="validation",
                                              to_tensor=True,
                                              pad=True, num_workers=1,
                                              shuffle=True, batch_size=2)

        assert_not_equals(train_batch_gen, None)
        assert_not_equals(test_batch_gen, None)
        assert_not_equals(validation_batch_gen, None)

        for n, batch in train_batch_gen:
            print(n)
            if n == 2:
                break

        for n, batch in test_batch_gen:
            print(n)
            if n == 2:
                break

        for n, batch in validation_batch_gen:
            print(n)
            if n == 2:
                break

