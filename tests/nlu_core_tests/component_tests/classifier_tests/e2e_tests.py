


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestToxic(unittest.TestCase):

    def test_toxic_model(self):
        df = nlu.load('en.classify.e2e',verbose=True).predict('You are so stupid')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print(df.columns)
        print(df['sentence'], df[['category','category_confidence']])

        df = nlu.load('e2e',verbose=True).predict('You are so stupid')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print(df.columns)
        print(df['sentence'], df[['category','category_confidence']])

if __name__ == '__main__':
    unittest.main()

