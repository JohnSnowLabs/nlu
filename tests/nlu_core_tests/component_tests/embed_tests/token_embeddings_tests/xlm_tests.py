


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestxlmEmbeddings(unittest.TestCase):

    def test_xlm(self):
        p  = nlu.load('xx.embed.xlm',verbose=True)
        df = p.predict("I love new embeds baby", output_level='token')
        for c in df.columns: print(df[c])



if __name__ == '__main__':
    unittest.main()

