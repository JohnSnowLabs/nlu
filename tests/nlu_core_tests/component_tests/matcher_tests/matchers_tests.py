import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class StemTests(unittest.TestCase):

    def test_stem(self):
        pipe = nlu.load('stem', verbose=True )
        df = pipe.predict('2020 was a crazy year but wait for October 1. 2020')
        for c in df.columns: print(df[c])

        #others to test :
        # 'en.match.pattern'
        # 'en.match.chunks'

    def test_pattern_matcher(self):
        pipe = nlu.load('match.pattern', verbose=True )
        df = pipe.predict('2020 was a crazy year but wait for October 1. 2020')
        for c in df.columns: print(df[c])


    def test_chunk_matcher(self):
        pipe = nlu.load('match.chunks', verbose=True )
        df = pipe.predict('2020 was a crazy year but wait for October 1. 2020')
        for c in df.columns: print(df[c])



if __name__ == '__main__':
    unittest.main()

