


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestSarcasm(unittest.TestCase):

    def test_sarcasm_model(self):
        # Get dataset "
        # todo test light pipe for 50k+
        # ! wget -N https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sarcasm/train-balanced-sarcasm.csv
        path = '/home/loan/Documents/freelancework/jsl/nlu/nlu_git/tests/datasets/train-balanced-sarcasm.csv'
        sarcasm_df = pd.read_csv(path)
        sarcasm_df['text'] = sarcasm_df['comment']
        print(len(sarcasm_df))
        # max 50k , 60K dead
        # count = int(len(sarcasm_df)/15)
        count = 50100
        print('using ', count,' Rows')
        #setting meta to true will output scores for keywords. Lower scores are better
        # Sentiment confidence is 2 because it sums the confidences of multiple sentences
        df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'].iloc[0:count])
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])

        print(df.columns)
        print(df['sentence'], df[['category','category_confidence']])



if __name__ == '__main__':
    unittest.main()

