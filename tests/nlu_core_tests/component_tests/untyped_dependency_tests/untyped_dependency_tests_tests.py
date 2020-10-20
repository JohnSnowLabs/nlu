


import unittest
from nlu import *

class TestDepUntyped(unittest.TestCase):

    def test_dependency_untyped_model(self):
        df = nlu.load('dep.untyped',verbose=True).predict('I love peanutbutter and jelly')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print(df.columns)
        print(df['sentence'], df[['category','category_confidence']])



if __name__ == '__main__':
    unittest.main()

