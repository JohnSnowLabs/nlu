import unittest
import pandas as pd
import nlu
import tests.nlu_hc_tests.secrets as sct

class MultiNerTests(unittest.TestCase):

    def test_multi_ner_pipe(self):

        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        # res = nlu.load('en.ner.diseases en.resolve_chunk.snomed.findings', verbose=True).predict(['The patient has cancer and high fever and will die next week.', ' She had a seizure.'], drop_irrelevant_cols=False, metadata=True, )

        data = ['The patient has cancer and high fever and will die next week.', ' She had a seizure.']
        res = nlu.load('en.med_ner.tumour en.med_ner.radiology en.med_ner.diseases en.ner.onto ', verbose=True).predict(data )


        for c in res :print(res[c])

        print(res)
if __name__ == '__main__':
    MultiNerTests().test_entities_config()


