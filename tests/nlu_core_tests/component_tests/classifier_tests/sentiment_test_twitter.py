import unittest
from nlu import *


class TestSentimentTwitter(unittest.TestCase):
    def test_sentiment_twitter_model(self):
        pipe = nlu.load('sentiment.twitter', verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'],
                          output_level='sentence')
        print(df.columns)
        for c in df.columns: print(df[c])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'],
                          output_level='document')
        print(df.columns)
        for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()
