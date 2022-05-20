import unittest

from nlu import *


class TestSentenceDetector(unittest.TestCase):
    def test_sentence_detector(self):
        pipe = nlu.load(
            "sentence_detector",
            verbose=True,
        )
        df = pipe.predict(
            "I like my sentences detected. Some like their sentences warm. Warm is also good.",
            output_level="sentence",
            drop_irrelevant_cols=False,
            metadata=True,
        )
        for c in df.columns:
            print(df[c])

    def test_sentence_detector_multi_lang(self):
        pipe = nlu.load(
            "xx.sentence_detector",
            verbose=True,
        )
        df = pipe.predict(
            "I like my sentences detected. Some like their sentences warm. Warm is also good.",
            output_level="sentence",
            drop_irrelevant_cols=False,
            metadata=True,
        )
        for c in df.columns:
            print(df[c])

    def test_sentence_detector_pragmatic(self):
        pipe = nlu.load(
            "sentence_detector.pragmatic",
            verbose=True,
        )
        df = pipe.predict(
            "I like my sentences detected. Some like their sentences warm. Warm is also good.",
            output_level="sentence",
            drop_irrelevant_cols=False,
            metadata=True,
        )
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
