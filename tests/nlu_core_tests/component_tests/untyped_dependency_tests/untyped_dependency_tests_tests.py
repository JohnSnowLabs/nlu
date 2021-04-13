


import unittest
from nlu import *

class TestDepUntyped(unittest.TestCase):

    def test_dependency_untyped_model(self):
        # This test takes too much ram on standard github actions machine
        return
        df = nlu.load('dep.untyped',verbose=True).predict('I love peanutbutter and jelly', output_level='document')
        for c in df.columns: print(df[c])


        df = nlu.load('dep.untyped',verbose=True).predict('I love peanutbutter and jelly', output_level='sentence')
        for c in df.columns: print(df[c])

        df = nlu.load('dep.untyped',verbose=True).predict('I love peanutbutter and jelly', output_level='token')
        for c in df.columns: print(df[c])






if __name__ == '__main__':
    unittest.main()

