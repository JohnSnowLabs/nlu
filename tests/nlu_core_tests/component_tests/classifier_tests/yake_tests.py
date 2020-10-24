import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestYake(unittest.TestCase):

    def test_yake_model(self):
        #setting meta to true will output scores for keywords. Lower scores are better
        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper', metadata=True)
        print(df.columns)
        print(df)
        print(df[['keywords', 'keywords_score_confidence']])

        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper', output_level='token')
        print(df.columns)
        print(df[['keywords', 'keywords_score_confidence']])
        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper', output_level='chunk')
        print(df.columns)
        print(df[['keywords', 'keywords_score_confidence']])
        #Column name of confidence changed if yake at same or not at same output level!
        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper', output_level='document')
        print(df.columns)
        print(df[['keywords', 'keywords_confidence']])

if __name__ == '__main__':
    unittest.main()

