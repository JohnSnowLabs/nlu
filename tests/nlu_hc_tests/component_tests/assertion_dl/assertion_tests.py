import unittest
import tests.secrets as sct
import nlu
import nlu.pipe.pipe_component
from sparknlp.annotator import *


class AssertionTests(unittest.TestCase):

    def test_assertion_dl_model(self):
        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET)

        # data = 'Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain'
        # data = """Miss M. is a 67-year-old lady, with past history of COPD and Hypertension, presents with a 3-weeks history of a lump in her right Breast. The lump appeared suddenly, also painful. 5 days ago, another lump appeared in her right axilla. On examination a 2 x 3 cm swelling was seen in the right Breast. It was firm and also non-tender and immobile. There was no discharge. Another 1x1 cm circumferential swelling was found in the right Axilla, which was freely mobile and also tender. Her family history is remarkable for Breast cancer  (mother), cervical cancer (maternal grandmother), heart disease (father), COPD (Brother), dementia (Grandfather), diabetes (Grandfather), and CHF (Grandfather)."""
        # res = nlu.load('en.assert.healthcare', verbose=True).predict(data, metadata=True)  # .predict(data)
        data = 'Miss M. is a 67-year-old lady, with past history of COPD and Hypertension, ' \
               'presents with a 3-weeks history of a lump in her right Breast. ' \
               'The lump appeared suddenly, also painful. 5 days ago, another lump appeared in her right axilla.' \
               ' On examination a 2 x 3 cm swelling was seen in the right Breast.' \
               ' It was firm and also non-tender and immobile. There was no discharge. ' \
               'Another 1x1 cm circumferential swelling was found in the right Axilla, ' \
               'which was freely mobile and also tender.' \
               ' Her family history is remarkable for Breast cancer (mother), ' \
               'cervical cancer (maternal grandmother), heart disease (father), ' \
               'COPD (Brother), dementia (Grandfather), diabetes (Grandfather), and CHF (Grandfather).'

        res = nlu.load('en.assert.biobert', verbose=True).predict(data, metadata=True)
        print(res.columns)
        for c in res:
            print(res[c])
        print(res)



if __name__ == '__main__':
    AssertionTests().test_assertion_dl_model()
