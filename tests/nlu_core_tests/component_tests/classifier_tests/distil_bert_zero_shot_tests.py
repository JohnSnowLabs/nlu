import unittest

from nlu import *


class TestDistilBertZeroShotClassifier(unittest.TestCase):
    def test_distil_bert_zero_shot_classifier(self):
        pipe = nlu.load("distilbert_base_zero_shot_classifier_uncased_mnli", verbose=True)
        df = pipe.predict(
            ["I loved this movie when I was a child."],
            output_level="sentence",
        )
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
