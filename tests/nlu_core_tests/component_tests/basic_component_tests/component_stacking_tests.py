import unittest
import nlu
import pandas as pd
import numpy as np

class ComponentStackigntests(unittest.TestCase):

    def test_sentiment_stack(self):
        # df = nlu.load('sentiment elmo',verbose=True).predict('Hello world', output_level='document')
        # df = nlu.load('sentiment elmo',verbose=True).predict('Hello world', output_level='sentence')

        df = nlu.load('sentiment elmo',verbose=True).predict('Hello world', output_level='token')
        for c in df.columns: print(df[c])



    def test_emotion_stack(self):
        # df = nlu.load('sentiment elmo',verbose=True).predict('Hello world', output_level='document')
        # df = nlu.load('sentiment elmo',verbose=True).predict('Hello world', output_level='sentence')

        df = nlu.load('emotion elmo',verbose=True).predict('Hello world', output_level='token')
        for c in df.columns:print(df[c])


    def test_sarcasm_stack(self):
        # df = nlu.load('sentiment elmo',verbose=True).predict('Hello world', output_level='document')
        # df = nlu.load('sentiment elmo',verbose=True).predict('Hello world', output_level='sentence')

        df = nlu.load('sarcasm elmo',verbose=True).predict('Hello world', output_level='token')
        for c in df.columns:print(df[c])
        # print(df['sarcasm'])

    #
    # def test_component_stack(self):
    #     # pos
    #
    #     pipe = nlu.load('sentiment bert', verbose = True ) #  bert
    #     preds = pipe.predict('Helo world!')

    #     # pos bert sent OK but whenevers ent not last we have troublz
    #     # pipe = nlu.load('sentiment emotion') #  bert
    #     # preds = pipe.predict('Helo world!')
    #     #
    #     # pipe = nlu.load('pos emotion') #  bert
    #     # preds = pipe.predict('Helo world!')
    #     #
    #     # pipe = nlu.load('emotion pos sentiment') #  bert
    #     # preds = pipe.predict('Helo world!')
    #     #
    #     # pipe = nlu.load('emotion pos sentiment bert') #  bert
    #     # preds = pipe.predict('Helo world!')
    #
    #     print(preds.columns)
    #     print(preds)
    #
