


import unittest
from nlu import *

class PretrainedPipeTests(unittest.TestCase):

    def simple_pretrained_pipe_tests(self):
        df = nlu.load('ner.onto',verbose=True).predict('I love peanutbutter and jelly')
        for c in df.columns: print(df[c])

if __name__ == '__main__':
    unittest.main()

