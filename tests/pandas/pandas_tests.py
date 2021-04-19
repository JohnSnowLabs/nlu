import unittest
import pandas as pd
import numpy as np
import numpy as np
import nlu
import sparknlp
import pyspark

class PandasTests(unittest.TestCase):
    def test_modin(self):
        # ## works with RAY and DASK backends
        df_path = '/home/ckl/old_home/Documents/freelance/jsl/nlu/nlu4realgit/tests/datasets/covid/covid19_tweets.csv'
        pdf = pd.read_csv(df_path).iloc[:10]
        secrets_json_path = '/home/ckl/old_home/Documents/freelance/jsl/nlu/nlu4realgit/tests/nlu_hc_tests/spark_nlp_for_healthcare.json'



        # test 1 series chunk
        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical resolve.icd10pcs',verbose=True).predict(pdf.text.iloc[0], output_level='chunk')
        # for c in res.columns:print(res[c])

        # Test longer series chunk
        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical resolve.icd10pcs',verbose=True).predict(pdf.text.iloc[0:10], output_level='chunk')

        # Test df with text col chunk

        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical', verbose=True).predict(pdf.text.iloc[:10], output_level='document')
        # for c in res.columns:print(res[c])

        #en.resolve_chunk.icd10cm.clinical
        res = nlu.auth(secrets_json_path).load('en.resolve_chunk.icd10cm.clinical',verbose=True).predict(pdf.text[0:7], output_level='chunk')
        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical resolve.icd10pcs',verbose=True).predict(pdf.text[0:7], output_level='chunk')
        for c in res.columns:print(res[c])



        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical resolve.icd10pcs',verbose=True).predict(pdf.text[0:7], output_level='sentence')
        # for c in res.columns:print(res[c])



if __name__ == '__main__':
    PandasTests().test_entities_config()
