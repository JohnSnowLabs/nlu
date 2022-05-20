import unittest

from nlu import *


class TestE2E(unittest.TestCase):
    def test_e2e_model(self):
        df = nlu.load("en.classify.e2e", verbose=True).predict(
            "You are so stupid", output_level="document"
        )

        for c in df.columns:
            print(df[c])

        df = nlu.load("e2e", verbose=True).predict(
            "You are so stupid", output_level="sentence"
        )
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
