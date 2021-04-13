


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestMultipleEmbeddings(unittest.TestCase):

    def test_multiple_embeddings(self):
        df = nlu.load('bert en.embed.bert.small_L8_512 en.embed.bert.small_L8_512 en.embed.bert.small_L8_128  electra en.embed.bert.small_L10_128 en.embed.bert.small_L4_128',verbose=True).predict('Am I the muppet or are you the muppet?')
        for c in df.columns: print(df[c])





if __name__ == '__main__':
    unittest.main()

