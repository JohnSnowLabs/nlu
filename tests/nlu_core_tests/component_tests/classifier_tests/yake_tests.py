import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestYake(unittest.TestCase):


    def test_yake_model(self):
        # THIS IS STILL A PIPE!
        df = nlu.load('yake',verbose=True).predict('What a wonderful day! Arnold schwanenegger is the Terminator and he wants to get to the American chopper')
        print(df)
        print(df.columns)


if __name__ == '__main__':
    unittest.main()

