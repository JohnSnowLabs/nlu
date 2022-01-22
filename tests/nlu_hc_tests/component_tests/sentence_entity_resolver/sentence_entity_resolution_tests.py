import unittest
import pandas as pd
import nlu
import tests.secrets as sct
from sparknlp.annotator import BertSentenceEmbeddings
from tests.test_utils import *

class SentenceResolutionTests(unittest.TestCase):
    def test_assertion_dl_model(self):


        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        # b = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')

        # todo en.ner.ade Error not accessable in 2.7.6??
        s1='The patient has COVID. He got very sick with it.'
        s2='Peter got the Corona Virus!'
        s3='COVID 21 has been diagnosed on the patient'
        data = [s1,s2,s3]
        # en.resolve_sentence.icd10cm
        resolver_ref = 'en.resolve.icd10cm.augmented_billable'
        res = nlu.load(f'en.med_ner.diseases {resolver_ref}', verbose=True).predict(data, drop_irrelevant_cols=False, metadata=True)

        # res = nlu.load('en.ner.anatomy', verbose=True).predict(['The patient has cancer and a tumor and high fever and will die next week. He has pain in his left food and right upper brain', ' She had a seizure.'], drop_irrelevant_cols=False, metadata=True)
        print(res.columns)
        for c in res :
            print(c)
            print(res[c])

        print(res)


if __name__ == '__main__':
    SentenceResolutionTests().test_entities_config()


