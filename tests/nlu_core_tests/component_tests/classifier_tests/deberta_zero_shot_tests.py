import unittest

from nlu import *


class TestDeBertaZeroShotClassifier(unittest.TestCase):
    def test_bert_zero_shot_classifier(self):
        pipe = nlu.load("en.deberta.zero_shot_classifier")
        df = pipe.predict(["I loved this movie when I was a child."])
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
