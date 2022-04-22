import unittest
from nlu import *
import tests.secrets as sct


class TestAuthentification(unittest.TestCase):

    def test_auth_via_file(self):

        secrets_json_path = '/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit/tests/nlu_hc_tests/spark_nlp_for_healthcare.json'
        res = nlu.auth(secrets_json_path).load('en.med_ner.diseases',verbose=True).predict("He has cancer")
        # res = nlu.load('en.med_ner.diseases',verbose=True).predict("He has cancer")
        for c in res.columns:print(res[c])
    def test_auth_miss_match(self):
        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        # JSL_SECRET            = sct.JSL_SECRET_3_4_2
        JSL_SECRET            = sct.JSL_SECRET
        res = nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET).load('en.med_ner.diseases').predict("He has cancer")
        for c in res.columns:print(res[c])

if __name__ == '__main__':
    unittest.main()

