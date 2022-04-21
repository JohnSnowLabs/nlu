import unittest

from nlu import *


class TestBertTokenEmbeddings(unittest.TestCase):
    def test_bert_model(self):
        df = nlu.load("bert", verbose=True).predict(
            "Am I the muppet or are you the muppet?"
        )
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])
        for c in df.columns:
            print(df[c])

    def test_multiple_bert_models(self):
        df = nlu.load(
            "en.embed.bert.small_L4_128 en.embed.bert.small_L2_256", verbose=True
        ).predict("No you are the muppet!")
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
