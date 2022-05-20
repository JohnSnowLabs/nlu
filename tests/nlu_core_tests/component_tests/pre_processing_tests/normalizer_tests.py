import unittest

from nlu import *


class TestNormalize(unittest.TestCase):
    def test_norm_pipe(self):
        pipe = nlu.load("norm", verbose=True)
        df = pipe.predict(
            "HELLO WORLD! How are YOU!?!@",
            output_level="sentence",
            drop_irrelevant_cols=False,
            metadata=True,
        )
        for c in df.columns:
            print(df[c])

        pipe["normalizer"].setLowercase(True)

        df = pipe.predict("HELLO WORLD! How are YOU!@>?!@")
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
