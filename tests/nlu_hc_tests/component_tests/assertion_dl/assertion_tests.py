import unittest
import pandas as pd
import nlu

class AssertionTests(unittest.TestCase):





    def test_assertion_dl_model(self):

        SPARK_NLP_LICENSE     = ''
        AWS_ACCESS_KEY_ID     = ''
        AWS_SECRET_ACCESS_KEY = ''
        JSL_SECRET            = ''

        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        import sparknlp_jsl
        ## TODO ASSERT may have different EMBED requirements than the NER model!
        res = nlu.load('ner en.ner.risk_factors en.assert en.resolve_chunk.athena_conditions', verbose=True).predict(['The patient has cancer and high fever and will die next week.', ' She had a seizure.'], drop_irrelevant_cols=False, metadata=True)

        for c in res :
            print(res[c])

        print(res)
if __name__ == '__main__':
    AssertionTests().test_entities_config()


