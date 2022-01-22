import unittest
import pandas as pd
import nlu
import sparknlp_jsl
from sparknlp.annotator import BertSentenceEmbeddings
from sparknlp_jsl.annotator import *
import tests.secrets as sct


class LicensedClassifierTests(unittest.TestCase):
    def test_LicensedClassifier(self):

        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET


        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        # b = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')
        # m = RelationExtractionModel().pretrained("posology_re")
#
        # res = nlu.load('en.ner.deid.augmented  en.de_identify', verbose=True).predict('DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin', return_spark_df=True)

        res = nlu.load('en.classify.ade.conversational', verbose=True).predict('DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin')


        print(res)
        for c in res :
            print(c)
            print(res[c])

if __name__ == '__main__':
    LicensedClassifierTests().test_LicensedClassifier()

