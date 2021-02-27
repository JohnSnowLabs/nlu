import unittest
import pandas as pd
import nlu
import sparknlp_jsl
from sparknlp.annotator import BertSentenceEmbeddings

class AssertionTests(unittest.TestCase):
    def test_assertion_dl_model(self):

        SPARK_NLP_LICENSE     = ''
        AWS_ACCESS_KEY_ID     = ''
        AWS_SECRET_ACCESS_KEY = ''
        JSL_SECRET            = ''
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        # b = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')

        res = nlu.load('en.resolve_sentence', verbose=True).predict(['The patient has cancer and high fever and will die next week.', ' She had a seizure.'], drop_irrelevant_cols=False, metadata=True)

        for c in res :
            print(c)
            print(res[c])

        print(res)
if __name__ == '__main__':
    AssertionTests().test_entities_config()

