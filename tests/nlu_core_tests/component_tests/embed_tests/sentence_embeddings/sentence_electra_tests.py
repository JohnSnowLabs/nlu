


import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestElectraSentenceEmbeddings(unittest.TestCase):

    def test_electra_sentence_embeds(self):
        df = nlu.load('embed_sentence.electra',verbose=True).predict('Am I the muppet or are you the muppet?')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])
        print(df.columns)
        print(df)
        print(df['embed_sentence_electra_embeddings'])

        df = nlu.load('en.embed_sentence.electra',verbose=True).predict('Am I the muppet or are you the muppet?')
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])
        print(df.columns)
        print(df)
        print(df['en_embed_sentence_electra_embeddings'])


        # df = nlu.load('en.embed.bert.small_L4_128', verbose=True).predict("No you are the muppet!")
        # print(df.columns)
        # print(df)
        # print(df['bert_embeddings'])




if __name__ == '__main__':
    unittest.main()

