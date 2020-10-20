


import unittest
from nlu import *

class PretrainedPipeTests(unittest.TestCase):

    def simple_pretrained_pipe_tests(self):
        df = nlu.load('ner.bert',verbose=True).predict('I love peanutbutter and jelly')
        df = nlu.load('ner.onto',verbose=True).predict('I love peanutbutter and jelly')

        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print(df.columns)
        print(df)


if __name__ == '__main__':
    unittest.main()

