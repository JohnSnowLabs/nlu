


import unittest
from nlu import *

class TestSentiment(unittest.TestCase):

    def test_sentiment_model(self):
        df = nlu.load('sentiment',verbose=True).predict('Women belong in the kitchen') # sorry we dont mean it
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print(df.columns)
        print(df['sentence'], df[['sentiment','sentiment_confidence']])



if __name__ == '__main__':
    unittest.main()

