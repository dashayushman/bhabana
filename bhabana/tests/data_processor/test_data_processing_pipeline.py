from nose.tools import *
from bhabana.processing import *
from bhabana.utils import constants
from bhabana.utils.data_utils import tokenize


class TestDataProcessor:

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

    def test_pipeline_tokenizer_seq2id(self):
        corpus = "Hallo, ich bin Dash. Ich bin Student und möchte Spaß haben"
        test_text = "Hallo, blah blah"
        self.create_vocab(corpus, "de")
        tokenizer = Tokenizer(lang="de", mode="word")
        seq2id = Seq2Id(self.w2i, seq_begin=True, seq_end=True)
        pipeline = [tokenizer, seq2id]
        dpp = DataProcessingPipeline(pipeline, name="my_pipeline",
                                          add_to_output=False)
        out = dpp.run(test_text)
        assert_not_equals(out, None)
        assert_equals(len(out), 6)
        assert_equals(out[0], constants.BOS)
        assert_equals(out[1], self.w2i["Hallo"])
        assert_equals(out[2], self.w2i[","])
        assert_equals(out[3], constants.UNK)
        assert_equals(out[4], constants.UNK)
        assert_equals(out[5], constants.EOS)
        assert_equals(dpp.name, "my_pipeline")
        assert_equals(dpp._get_name(), "Tokenizer_Seq2Id")
