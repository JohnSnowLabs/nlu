


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestNGram(unittest.TestCase):

    def test_ngram(self):
        example_text =  ["A person like Jim or Joe",
                     "An organisation like Microsoft or PETA",
                     "A location like Germany",
                     "Anything else like Playstation",
                     "Person consisting of multiple tokens like Angela Merkel or Donald Trump",
                     "Organisations consisting of multiple tokens like JP Morgan",
                     "Locations consiting of multiple tokens like Los Angeles",
                     "Anything else made up of multiple tokens like Super Nintendo",]


        print('OUTPUT LEVEL TOKEN')
        n_df = nlu.load('ngram').predict(example_text, output_level='token')
        print(n_df.columns)
        print(n_df)

        print('OUTPUT LEVEL CHUNK')
        n_df = nlu.load('ngram',verbose=True).predict(example_text, output_level='chunk')
        print(n_df.columns)
        print(n_df)

        print('OUTPUT LEVEL SENTENCE')
        n_df = nlu.load('ngram',verbose=True).predict(example_text, output_level='sentence')
        print(n_df.columns)
        print(n_df)


        print('OUTPUT LEVEL DOCUMENT')
        n_df = nlu.load('ngram',verbose=True).predict(example_text, output_level='document')
        print(n_df.columns)
        print(n_df)

if __name__ == '__main__':
    unittest.main()

