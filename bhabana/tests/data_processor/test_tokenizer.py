from bhabana.utils import constants

from nose.tools import *
from bhabana.processing import Tokenizer


class TestTokenizer:
    text = ["Hallo, ich bin Dash", "I am ayushman"]
    languages = ["de"]

    def test_semhash_tokenizer(self):
        text = "Hallo, ich"
        tokenizer = Tokenizer(lang="de", mode="semhash")
        semhash_tokens = tokenizer.process(text)
        assert_equals(len(semhash_tokens), 3)
        assert_equals(semhash_tokens[0][0], "#Ha")
        assert_equals(semhash_tokens[0][1], "Hal")
        assert_equals(semhash_tokens[0][2], "all")
        assert_equals(semhash_tokens[0][3], "llo")
        assert_equals(semhash_tokens[0][4], "lo#")

        assert_equals(semhash_tokens[1][0], "#,#")

        assert_equals(semhash_tokens[2][0], "#ic")
        assert_equals(semhash_tokens[2][1], "ich")
        assert_equals(semhash_tokens[2][2], "ch#")

    def test_char_tokenizer(self):
        text = "Hallo, ich"
        tokenizer = Tokenizer(lang="de", mode="char")
        char_tokens = tokenizer.process(text)
        assert_equals(len(char_tokens), 10)
        assert_equals(char_tokens[6], constants.SPACE_WORD)
        assert_equals(char_tokens[0], 'H')
        assert_equals(char_tokens[-1], 'h')

    def test_word_tokenizer_de(self):
        text = "Hallo, ich"
        tokenizer = Tokenizer(lang="de", mode="word")
        semhash_tokens = tokenizer.process(text)
        assert_equals(len(semhash_tokens), 3)
        assert_equals(semhash_tokens[0], "Hallo")
        assert_equals(semhash_tokens[1], ',')
        assert_equals(semhash_tokens[2], 'ich')

    def test_word_tokenizer_en(self):
        text = "Hallo, ich"
        tokenizer = Tokenizer(lang="en", mode="word")
        semhash_tokens = tokenizer.process(text)
        assert_equals(len(semhash_tokens), 3)
        assert_equals(semhash_tokens[0], "hallo")
        assert_equals(semhash_tokens[1], ',')
        assert_equals(semhash_tokens[2], 'ich')


