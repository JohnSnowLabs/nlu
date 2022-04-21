import unittest

from nlu import *


class TestLem(unittest.TestCase):
    def test_stopwords_pipe(self):
        pipe = nlu.load("stopwords", verbose=True)
        df = pipe.predict(
            "HELLO WORLD! How are YOU!?!@",
            output_level="sentence",
            drop_irrelevant_cols=False,
            metadata=True,
        )
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
