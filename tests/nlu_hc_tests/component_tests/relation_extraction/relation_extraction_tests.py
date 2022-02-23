import unittest
import pandas as pd
import nlu
import sparknlp_jsl
from sparknlp.annotator import BertSentenceEmbeddings
from sparknlp_jsl.annotator import *
import tests.secrets as sct

class RelationExtractionTests(unittest.TestCase):
    def test_relation_extraction(self):

        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET


        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        # res = nlu.load('en.ner.posology en.extract_relation.drug_drug_interaction', verbose=True).predict('The patient got cancer in my foot and damage in his brain but we gave him 50G of  and 50mg Penicilin and this helped is brain injury after 6 hours. 1 Hour after the penicilin, 3mg Morphium was administred which had no problems with the Penicilin', return_spark_df=True)
        s1='The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also given 1 unit of Metformin daily. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day.'
        data =[s1]
        # res = nlu.load('med_ner.posology relation.drug_drug_interaction', verbose=True).predict(data, drop_irrelevant_cols=False, metadata=True, )
        res = nlu.load('relation.drug_drug_interaction', verbose=True).predict(data, drop_irrelevant_cols=False, metadata=True, )


        for c in res :
            print(c)
            print(res[c])

        # print(res)
if __name__ == '__main__':
    RelationExtractionTests().test_entities_config()

