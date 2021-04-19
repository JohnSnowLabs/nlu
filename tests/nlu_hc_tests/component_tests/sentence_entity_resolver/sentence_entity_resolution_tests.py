import unittest
import pandas as pd
import nlu
import tests.nlu_hc_tests.secrets as sct
from sparknlp.annotator import BertSentenceEmbeddings
from tests.test_utils import *

class AssertionTests(unittest.TestCase):
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
        #TODO Not correct
        resolver_ref = 'en.resolve_sentence.icd10cm.augmented_billable'
        res = nlu.load(f'en.ner.diseases {resolver_ref}', verbose=True).predict(data, drop_irrelevant_cols=False, metadata=True)

        # res = nlu.load('en.ner.anatomy', verbose=True).predict(['The patient has cancer and a tumor and high fever and will die next week. He has pain in his left food and right upper brain', ' She had a seizure.'], drop_irrelevant_cols=False, metadata=True)
        print(res.columns)
        for c in res :
            print(c)
            print(res[c])

        print(res)
    def test_quick(self):
        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET

        s3='The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also given 1 unit of Metformin daily. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day and then he got COVID and herpes.'
        s2='What is the capital of Germany>?'
        s1 = 'What is the most spoken language in France?'
        data =[s1,s2,s3]
        # TODO COL OVERLAPS!! --> Each annotator (or at least clasisfiers/overlaps) will be named with <type>@<nlu_ref_leaf> === <type>@identifier(nlu_ref_leaf)
        # ref = 'en.extract_relation.bodypart.problem' # TODO BAD
        # ref = 'ner'

        # ref = 'en.med_ner.diseases' # TODO AUTH BUG???
        # res = nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET).load(f'en.ner.diseases {ref}', verbose=True).predict(data)
        df = get_sample_pdf_with_extra_cols_and_entities()
        # res = nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET).load(f' {ref}', verbose=True).predict(df)
        # res = nlu.load(f' {ref}', verbose=True).predict(df, output_level='sentence')


        ref = '/home/loan/tmp/nlu_models_offline_test/analyze_sentiment_en_3.0.0_3.0_1616544471011'
        res = nlu.load(path=f'{ref}', verbose=True).predict(df, output_level='sentence')

        for c in res : print(res[c])


if __name__ == '__main__':
    AssertionTests().test_entities_config()


