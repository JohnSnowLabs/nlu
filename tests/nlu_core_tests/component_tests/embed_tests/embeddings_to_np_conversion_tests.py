import unittest

from nlu import *


class TestEmbeddingsConversion(unittest.TestCase):
    def test_word_embeddings_conversion(self):
        df = nlu.load("bert", verbose=True).predict("How are you today")
        for c in df.columns:
            print(df[c])

    def test_sentence_embeddings_conversion(self):
        df = nlu.load("embed_sentence.bert", verbose=True).predict("How are you today")
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
