


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestToxic(unittest.TestCase):

    def test_toxic_model(self):

        pipe = nlu.load('toxic',verbose=True)
        df = pipe.predict(['You stupid man', 'You stupid woman'], output_level='sentence')
        print(df)
        print(df.columns)
        print(df['sentence'], df[['toxic','toxic_confidence']])
        df = pipe.predict(['You stupid man', 'You stupid woman'], output_level='document')
        self.assertIsInstance(df.iloc[0]['toxic'],str )

        print(df)
        print(df.columns)
        print(df['document'], df[['toxic','toxic_confidence']])
        self.assertIsInstance(df.iloc[0]['toxic'], str)



if __name__ == '__main__':
    unittest.main()

