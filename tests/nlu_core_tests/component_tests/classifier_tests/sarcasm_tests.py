import unittest

from nlu import *


class TestSarcasm(unittest.TestCase):
    def test_sarcasm_model(self):
        pipe = nlu.load("sarcasm", verbose=True)
        df = pipe.predict(
            ["I love pancaces. I hate Mondays", "I love Fridays"],
            output_level="sentence",
        )
        for c in df.columns:
            print(df[c])
        df = pipe.predict(
            ["I love pancaces. I hate Mondays", "I love Fridays"],
            output_level="document",
        )
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
