import unittest

import nlu
from nlu import *


class TestDistilBertZeroShotClassifier(unittest.TestCase):
    def test_distil_bert_zero_shot_classifier(self):

        pipe = nlu.load("en.distilbert.zero_shot_classifier", verbose=True)
        df = pipe.predict(["I have a problem with my iphone that needs to be resolved asap!!"],
                          output_level="sentence",
                          )
        for c in df.columns:
            print(df[c])

        # Turkish Models and difference examples.

        pipe = nlu.load("tr.distilbert.zero_shot_classifier.multinli", verbose=True)
        df = pipe.predict(['Dolar yükselmeye devam ediyor.'], output_level="sentence", )
        for c in df.columns:
            print(df[c])

        pipe = nlu.load("tr.distilbert.zero_shot_classifier.allnli", verbose=True)
        df = pipe.predict(['Senaryo çok saçmaydı, beğendim diyemem.'], output_level="sentence", )
        for c in df.columns:
            print(df[c])

        pipe = nlu.load("tr.distilbert.zero_shot_classifier.snli", verbose=True)
        df = pipe.predict(
            ['Senaryo çok saçmaydı, beğendim diyemem.'],
            output_level="sentence",
        )

        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()