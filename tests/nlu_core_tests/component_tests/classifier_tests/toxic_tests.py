


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestToxic(unittest.TestCase):

    def test_toxic_model(self):
        # nlu.load('en.ner.dl.bert').predict("I like Angela Merkel")
        pipe = nlu.load('toxic',verbose=True)
        data = ['You are so dumb you goofy dummy', 'You stupid person with an identity that shall remain unnamed, such a filthy identity that you have go to a bad place you person!']
        df = pipe.predict(data, output_level='sentence')
        for c in df.columns: print(df[c])

        df = pipe.predict(data, output_level='document',metadata=True)
        for c in df.columns: print(df[c])



if __name__ == '__main__':
    unittest.main()


