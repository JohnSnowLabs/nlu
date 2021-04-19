


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestE2E(unittest.TestCase):

    def test_e2e_model(self):
        df = nlu.load('en.classify.e2e',verbose=True).predict('You are so stupid', output_level='document')

        for c in df.columns: print(df[c])

        df = nlu.load('e2e',verbose=True).predict('You are so stupid', output_level='sentence')
        for c in df.columns: print(df[c])





if __name__ == '__main__':
    unittest.main()

