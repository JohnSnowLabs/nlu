


import unittest
from nlu import *

class TestSentiment(unittest.TestCase):
    def test_sentiment_model(self):
        pipe = nlu.load('sentiment',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])

    def test_sentiment_imdb_model(self):
        pipe = nlu.load('sentiment.twitter',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'])#, output_level='document',drop_irrelevant_cols=False, metadata=True, )
        print(df.columns)
        for c in df.columns: print(df[c])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        print(df.columns)
        for c in df.columns: print(df[c])



    def test_sentiment_twitter_model(self):
        pipe = nlu.load('sentiment.imdb',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence')
        print(df.columns)
        for c in df.columns: print(df[c])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        print(df.columns)
        for c in df.columns: print(df[c])


    def test_sentiment_detector_model(self):
        pipe = nlu.load('sentiment.imdb',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence')
        for c in df.columns: print(df[c])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        for c in df.columns: print(df[c])

    def test_sentiment_vivk_model(self):
        pipe = nlu.load('sentiment.vivekn',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence')
        print(df.columns)
        for c in df.columns: print(df[c])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()

