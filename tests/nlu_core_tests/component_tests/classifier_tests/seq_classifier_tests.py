import unittest
from nlu import *


class TestSeqClassifier(unittest.TestCase):
    def test_sentiment_model(self):
        seq_pipe = nlu.load('en.classify.roberta.imdb')
        df = seq_pipe.predict('This movie was fucking asweomse!')
        for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()
