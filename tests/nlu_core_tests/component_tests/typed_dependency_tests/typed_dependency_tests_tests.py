


import unittest
from nlu import *

class TestDepTyped(unittest.TestCase):

    def test_dependency_typed_model(self):
        # This test takes too much ram on standard github actions machine
        return
        df = nlu.load('dep.typed',verbose=True).predict('I love peanutbutter and jelly', output_level='document')
        print("DOCUMENT")
        print(df.columns)
        print(df['document'], df[['dependency','pos']])
        print(df['document'], df[['labled_dependency','pos']])


        print("SENTENCE")
        df = nlu.load('dep.typed',verbose=True).predict('I love peanutbutter and jelly', output_level='sentence')
        print(df.columns)
        print(df['sentence'], df[['dependency','pos']])
        print(df['sentence'], df[['labled_dependency','pos']])

        print("TOKEN")
        df = nlu.load('dep.typed',verbose=True).predict('I love peanutbutter and jelly', output_level='token')
        print(df.columns)
        print(df['token'], df[['dependency','pos']])
        print(df['token'], df[['labled_dependency','pos']])


if __name__ == '__main__':
    unittest.main()

