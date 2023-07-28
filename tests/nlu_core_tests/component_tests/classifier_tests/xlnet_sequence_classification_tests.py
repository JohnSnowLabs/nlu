import unittest

import nlu


class TestXlnetSequenceClassifier(unittest.TestCase):

    def test_xlnet_sequence_classifier(self):
        pipe = nlu.load("en.classify.ag_news.xlnet", verbose=True)
        data = "Disney Comics was a comic book publishing company operated by The Walt Disney Company which ran from 1990 to 1993."
        df = pipe.predict([data], output_level="document")
        for c in df.columns:
            print(df[c])

        pipe = nlu.load("en.classify.imdb.xlnet", verbose=True)
        data = "I really liked that movie!"
        df = pipe.predict([data], output_level="document")
        for c in df.columns:
            print((df[c]))


if __name__ == "__main__":
    unittest.main()
