import unittest

from nlu import *


class TestdistilbertEmbeddings(unittest.TestCase):
    def test_distilbert(self):
        df = nlu.load("xx.embed.distilbert", verbose=True).predict(
            "Am I the muppet or are you the muppet?", output_level="token"
        )
        for c in df.columns:
            print(df[c])

    def test_NER(self):
        df = nlu.load("ner", verbose=True).predict(
            "Donald Trump from America and Angela Merkel from Germany are BFF"
        )
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
