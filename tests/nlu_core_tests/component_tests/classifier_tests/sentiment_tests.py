


import unittest
from nlu import *

class TestSentiment(unittest.TestCase):
    def test_sentiment_model(self):
        pipe = nlu.load('sentiment',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence')
        print(df.columns)
        print(df['sentence'], df[['sentiment','sentiment_confidence']])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        self.assertIsInstance(df.iloc[0]['sentiment'],str )
        print(df.columns)
        print(df['document'], df[['sentiment','sentiment_confidence']])
        self.assertIsInstance(df.iloc[0]['sentiment'], str)

if __name__ == '__main__':
    unittest.main()

