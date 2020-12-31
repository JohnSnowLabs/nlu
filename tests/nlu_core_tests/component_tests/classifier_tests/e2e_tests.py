


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestE2E(unittest.TestCase):

    def test_e2e_model(self):
        df = nlu.load('en.classify.e2e',verbose=True).predict('You are so stupid', output_level='document')

        print(df.columns)
        print(df['document'], df[['e2e_classes','e2e_confidences']])

        df = nlu.load('e2e',verbose=True).predict('You are so stupid', output_level='sentence')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print(df.columns)
        print(df['sentence'], df[['e2e_classes','e2e_confidences']])


    def test_quick(self):
        # pipe = nlu.load('embed_sentence.bert')
        # predictions = pipe.predict(get_sample_pdf(), output_level='document')
        # print(predictions)
        p  = '/home/loan/Documents/freelancework/jsl/KNOWLEDGE_GRAPH/papaers/test.csv'
        import pandas as pd
        df = pd.read_csv(p)
        # THIS CRASHES WITH USE LAST!!
        # multi_pipe = nlu.load('en.embed_sentence.electra embed_sentence.bert use', )
        # multi_pipe = nlu.load('en.embed_sentence.electra embed_sentence.bert use', )
        multi_pipe = nlu.load('use en.embed_sentence.electra embed_sentence.bert', )

        # res = multi_pipe.predict( get_sample_pdf(), output_level='document')
        res = multi_pipe.predict(df.Title, output_level='document')

        print(res)
        print(res.columns)






if __name__ == '__main__':
    unittest.main()

