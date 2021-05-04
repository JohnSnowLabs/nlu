


import unittest
from nlu import *

class TestDepTyped(unittest.TestCase):

    def test_dependency_typed_model(self):
        # This test takes too much ram on standard github actions machine
        df = nlu.load('dep.typed',verbose=True).predict('I love peanutbutter and jelly', output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])


        print("SENTENCE")
        df = nlu.load('dep.typed',verbose=True).predict('I love peanutbutter and jelly', output_level='sentence')
        for c in df.columns: print(df[c])

        print("TOKEN")
        df = nlu.load('dep.typed',verbose=True).predict('I love peanutbutter and jelly', output_level='token')
        for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()

