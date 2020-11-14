


import unittest
from nlu import *

class TestDepUntyped(unittest.TestCase):

    def test_dependency_untyped_model(self):
        # This test takes too much ram on standard github actions machine
        return
        df = nlu.load('dep.untyped',verbose=True).predict('I love peanutbutter and jelly', output_level='document')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])
        print("DOCUMENT")
        print(df.columns)
        print(df['document'], df[['dependency','pos']])



        df = nlu.load('dep.untyped',verbose=True).predict('I love peanutbutter and jelly', output_level='sentence')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print("SENTENCE")
        print(df.columns)
        print(df['sentence'], df[['dependency','pos']])

        df = nlu.load('dep.untyped',verbose=True).predict('I love peanutbutter and jelly', output_level='token')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])
        print("TOKEN")
        print(df.columns)
        print(df['token'], df[['dependency','pos']])





if __name__ == '__main__':
    unittest.main()

