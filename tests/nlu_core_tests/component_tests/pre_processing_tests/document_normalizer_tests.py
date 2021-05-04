import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestNormalize(unittest.TestCase):

    def test_document_normalizer_pipe(self):
        pipe = nlu.load('norm_document', verbose=True )
        data = '<!DOCTYPE html> <html> <head> <title>Example</title> </head> <body> <p>This is an example of a simple HTML page with one paragraph.</p> </body> </html>'
        df = pipe.predict(data,output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()

