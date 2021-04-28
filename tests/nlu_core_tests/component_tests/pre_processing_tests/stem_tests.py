import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestStem(unittest.TestCase):

    def test_stem_pipe(self):
        pipe = nlu.load('stem', verbose=True )
        df = pipe.predict('HELLO WORLD! How are YOU!?!@', output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()

