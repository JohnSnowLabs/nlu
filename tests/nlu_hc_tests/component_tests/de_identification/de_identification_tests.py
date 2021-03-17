import unittest
import pandas as pd
import nlu
import sparknlp_jsl
from sparknlp.annotator import BertSentenceEmbeddings
from sparknlp_jsl.annotator import *

class DeidentificationTests(unittest.TestCase):
    def test_deidentification(self):

        SPARK_NLP_LICENSE     = ''
        AWS_ACCESS_KEY_ID     = ''
        AWS_SECRET_ACCESS_KEY = ''
        JSL_SECRET            = ''
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        # b = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')
        # m = RelationExtractionModel().pretrained("posology_re")
#
        # res = nlu.load('en.ner.deid.augmented  en.de_identify', verbose=True).predict('DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin', return_spark_df=True)

        res = nlu.load('en.de_identify', verbose=True)#.predict('DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin', return_spark_df=True)
        # res = nlu.load('zh.segment_words pos', verbose=True)#.predict('DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin', return_spark_df=True)

        print('YOYOYO')

        res.show()
        for c in res.columns:
            print(c)
            res.select(c).show(truncate=False)
        # res = nlu.load('en.extract_relation', verbose=True).predict('The patient got cancer in my foot and damage in his brain')


        # for c in res :
        #     print(c)
        #     print(res[c])

        # print(res)
if __name__ == '__main__':
    DeidentificationTests().test_entities_config()

