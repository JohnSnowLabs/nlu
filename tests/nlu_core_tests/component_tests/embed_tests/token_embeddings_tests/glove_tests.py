import unittest

from nlu import *


class TestGloveTokenEmbeddings(unittest.TestCase):
    def test_glove_model(self):
        df = nlu.load("glove", verbose=True).predict(
            "Am I the muppet or are you the muppet?", output_level="token"
        )
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
