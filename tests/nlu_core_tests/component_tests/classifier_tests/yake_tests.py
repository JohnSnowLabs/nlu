import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestYake(unittest.TestCase):

    def test_yake_model(self):
        #setting meta to true will output scores for keywords. Lower scores are better
        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper', output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])
        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper', metadata=False)
        for c in df.columns: print(df[c])
        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper', output_level='token')
        for c in df.columns: print(df[c])
        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper', output_level='chunk')
        for c in df.columns: print(df[c])
        #Column name of confidence changed if yake at same or not at same output level!
        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper', output_level='document')
        for c in df.columns: print(df[c])

if __name__ == '__main__':
    unittest.main()

