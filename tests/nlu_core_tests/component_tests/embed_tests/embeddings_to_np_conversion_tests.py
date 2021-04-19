


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *
import numpy as np
class TestEmbeddingsConversion(unittest.TestCase):

    def test_word_embeddings_conversion(self):
        df = nlu.load('bert',verbose=True).predict('How are you today')
        for c in df.columns: print(df[c])
        
    def test_sentence_embeddings_conversion(self):
        df = nlu.load('embed_sentence.bert',verbose=True).predict('How are you today')
        for c in df.columns: print(df[c])



if __name__ == '__main__':
    unittest.main()

