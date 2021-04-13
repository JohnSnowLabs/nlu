from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
import unittest
# from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *
import tests.nlu_hc_tests.secrets as sct


class TestAuthentification(unittest.TestCase):

    def test_auth_via_file(self):
        secrets_json_path = '/home/ckl/Downloads/tmp/spark_nlp_for_healthcare.json'
        res = nlu.auth(secrets_json_path).load('en.med_ner.diseases',verbose=True).predict("He has cancer")
        for c in res.columns:print(res[c])
if __name__ == '__main__':
    unittest.main()

