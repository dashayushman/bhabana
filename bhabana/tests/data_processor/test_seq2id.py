from nose.tools import *
from bhabana.processing import Seq2Id
from bhabana.utils import constants
from bhabana.utils.data_utils import tokenize


# TODO: add tests for all languages
class TestSeq2Id:
    def setUp(self):
        self.w2i = {constants.PAD_WORD: constants.PAD,
               constants.UNK_WORD: constants.UNK,
               constants.BOS_WORD: constants.BOS,
               constants.EOS_WORD: constants.EOS,
               constants.SPACE_WORD: constants.SPACE}

    def create_vocab(self, corpus, lang):
        vocab = set(tokenize(corpus, lang=lang))
        id = 5
        for tok in vocab:
            if tok not in self.w2i:
                self.w2i[tok] = id
                id += 1

    def test_seq2id_bi(self):
        corpus = "Hallo, ich bin Dash. Ich bin Student und möchte Spaß haben"
        test_text_tokens = tokenize("Hallo, blah blah", lang="de") + [
            constants.PAD_WORD]
        self.create_vocab(corpus, "de")
        seq2id_bi = Seq2Id(self.w2i, seq_begin=True, seq_end=True)
        seq2id_bi_out = seq2id_bi.process(test_text_tokens)
        assert_true(seq2id_bi.is_valid_data(test_text_tokens)["is_valid"])
        assert_not_equals(seq2id_bi_out, None)
        assert_equals(len(seq2id_bi_out), 7)
        assert_equals(seq2id_bi_out[0], constants.BOS)
        assert_equals(seq2id_bi_out[1], self.w2i["Hallo"])
        assert_equals(seq2id_bi_out[2], self.w2i[","])
        assert_equals(seq2id_bi_out[3], constants.UNK)
        assert_equals(seq2id_bi_out[4], constants.UNK)
        assert_equals(seq2id_bi_out[5], constants.PAD)
        assert_equals(seq2id_bi_out[6], constants.EOS)

    def test_seq2id_front(self):
        corpus = "Hallo, ich bin Dash. Ich bin Student und möchte Spaß haben"
        test_text_tokens = tokenize("Hallo, blah blah", lang="de") + [
            constants.PAD_WORD]
        self.create_vocab(corpus, "de")
        seq2id_front = Seq2Id(self.w2i, seq_begin=True, seq_end=False)
        seq2id_front_output = seq2id_front.process(test_text_tokens)
        assert_true(seq2id_front.is_valid_data(test_text_tokens)["is_valid"])
        assert_not_equals(seq2id_front_output, None)
        assert_equals(len(seq2id_front_output), 6)
        assert_equals(seq2id_front_output[0], constants.BOS)
        assert_equals(seq2id_front_output[1], self.w2i["Hallo"])
        assert_equals(seq2id_front_output[2], self.w2i[","])
        assert_equals(seq2id_front_output[3], constants.UNK)
        assert_equals(seq2id_front_output[4], constants.UNK)
        assert_equals(seq2id_front_output[5], constants.PAD)




