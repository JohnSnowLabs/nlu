


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
        df = nlu.load('ngram', verbose=True).predict(example_text, output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])

        print('OUTPUT LEVEL CHUNK')
        df = nlu.load('ngram',verbose=True).predict(example_text, output_level='chunk')
        for c in df.columns: print(df[c])

        print('OUTPUT LEVEL SENTENCE')
        df = nlu.load('ngram',verbose=True).predict(example_text, output_level='sentence')
        for c in df.columns: print(df[c])

        print('OUTPUT LEVEL DOCUMENT')
        df = nlu.load('ngram',verbose=True).predict(example_text, output_level='document')
        for c in df.columns: print(df[c])

if __name__ == '__main__':
    unittest.main()

