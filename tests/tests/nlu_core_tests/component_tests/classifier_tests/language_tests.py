


import unittest
from nlu import *

class TestLanguage(unittest.TestCase):

    def test_language_model(self):
        import nlu
        import gc
        nlu.active_pipes.clear()
        gc.collect()
        df = nlu.load('lang',verbose=True).predict('I love peanutbutter and jelly')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print(df.columns)
        print(df[['language','language_confidence']])



if __name__ == '__main__':
    unittest.main()

