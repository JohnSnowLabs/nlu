


import unittest
from nlu import *

class TestQuestions(unittest.TestCase):

    def test_ner_model(self):
        df = nlu.load('questions',verbose=True).predict('Women belong in the kitchen') # sorry we dont mean it
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print(df.columns)
        print(df['sentence'], df[['category','category_confidence']])



if __name__ == '__main__':
    unittest.main()

