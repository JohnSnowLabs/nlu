import unittest
import tests.nlu_hc_tests.secrets as sct
import nlu
import nlu.pipe.pipe_components
from sparknlp.annotator import *
class DrugNormalizerTests(unittest.TestCase):


    def test_drug_normalizer(self):

        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)

        data = ["Agnogenic one half cup","adalimumab 54.5 + 43.2 gm","aspirin 10 meq/ 5 ml oral sol","interferon alfa-2b 10 million unit ( 1 ml ) injec","Sodium Chloride/Potassium Chloride 13bag"]
        res = nlu.load('norm_drugs').predict(data, output_level='document') # .predict(data)

        print(res.columns)
        for c in res :print(res[c])

        print(res)
if __name__ == '__main__':
    DrugNormalizerTests().test_entities_config()


