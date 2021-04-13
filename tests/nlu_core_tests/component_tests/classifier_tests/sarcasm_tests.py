


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestSarcasm(unittest.TestCase):
    def test_sarcasm_model(self):
        pipe = nlu.load('sarcasm',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence')
        for c in df.columns: print(df[c])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        for c in df.columns: print(df[c])

    
    #
    # def test_sarcasm_model_bench(self):
    #     # Get dataset "
    #     # todo test light pipe for 50k+
    #     # ! wget -N https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sarcasm/train-balanced-sarcasm.csv
    #     # path = '/home/loan/Documents/freelancework/jsl/nlu/nlu_git/tests/datasets/train-balanced-sarcasm.csv'
    #     path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/Musical_instruments_reviews.csv'
    #     sarcasm_df = pd.read_csv(path)
    #     # sarcasm_df['text'] = sarcasm_df['comment']
    #     # print(len(sarcasm_df))
    #     # max 50k , 60K dead
    #     # count = int(len(sarcasm_df)/15)
    #     # count = 50100
    #     # print('using ', count,' Rows')
    #     print(sarcasm_df.columns)
    #     #setting meta to true will output scores for keywords. Lower scores are better
    #     # Sentiment confidence is 2 because it sums the confidences of multiple sentences
    #     # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['reviewText'].iloc[0:100])
    #     df = nlu.load('bert',verbose=True).predict(sarcasm_df['reviewText'].iloc[0:100])
    #
    #     # df = nlu.load('en.classify.sarcasm',verbose=True).predict('How are you today')
    #
    #     # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])
    #
    #     print(df.columns)
    #     print(df['bert_embeddings'])


if __name__ == '__main__':
    unittest.main()

