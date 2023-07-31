import unittest

from nlu import *


class TestBertZeroShotClassifier(unittest.TestCase):
    def test_bert_zero_shot_classifier(self):
        pipe = nlu.load("en.bert.zero_shot_classifier", verbose=True)
        df = pipe.predict(
            ["I loved this movie when I was a child."],
            output_level="sentence"
        )
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
