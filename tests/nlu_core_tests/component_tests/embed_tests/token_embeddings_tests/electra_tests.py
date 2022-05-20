import unittest

from nlu import *


class TestElectraTokenEmbeddings(unittest.TestCase):
    def test_electra_model(self):

        df = nlu.load("bert electra ", verbose=True).predict(
            "Am I the muppet or are you the muppet?"
        )
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
