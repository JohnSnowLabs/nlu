


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestToxic(unittest.TestCase):

    def test_toxic_model(self):
        # nlu.load('en.ner.dl.bert').predict("I like Angela Merkel")
        pipe = nlu.load('toxic',verbose=True)
        data = ['You are so dumb you goofy dummy', 'You stupid person with an identity that shall remain unnamed, such a filthy identity that you have go to a bad place you person!']
        df = pipe.predict(data, output_level='sentence')
        print(df)
        print(df.columns)

        print(df['sentence'], df[['toxic_classes']])
        print(df['sentence'], df[['toxic_confidences']])
        df = pipe.predict(data, output_level='document',metadata=True)


        print(df)
        print(df.columns)
        print(df['document'], df[['toxic_obscene_confidence']])
        print(df['toxic_severe_toxic_confidence'], df[['toxic_insult_confidence']])
        print(df['toxic_toxic_confidence'], df[['toxic_obscene_confidence']])




if __name__ == '__main__':
    unittest.main()


