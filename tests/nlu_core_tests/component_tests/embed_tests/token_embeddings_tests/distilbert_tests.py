


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestdistilbertEmbeddings(unittest.TestCase):

    def test_distilbert(self):
        df = nlu.load('xx.embed.distilbert',verbose=True).predict('Am I the muppet or are you the muppet?', output_level='token')
        for c in df.columns: print(df[c])

    def test_NER(self):
        df = nlu.load('ner',verbose=True).predict('Donald Trump from America and Angela Merkel from Germany are BFF')
        for c in df.columns: print(df[c])





if __name__ == '__main__':
    unittest.main()

