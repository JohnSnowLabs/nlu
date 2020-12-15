import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestNormalize(unittest.TestCase):

    def test_spell(self):
        pipe = nlu.load('spell', verbose=True )
        df = pipe.predict('I liek penut butter and jellli')
        print(df)
        print(df.columns)
        print(df['checked'])

if __name__ == '__main__':
    unittest.main()

