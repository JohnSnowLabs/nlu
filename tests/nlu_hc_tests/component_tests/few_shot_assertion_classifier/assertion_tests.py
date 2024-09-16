# import tests.secrets as sct

import os
import sys

# sys.path.append(os.getcwd())
import unittest
import nlu

# os.environ["PYTHONPATH"] = "F:/Work/repos/nlu_new/nlu"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from johnsnowlabs import nlp, visual

# nlp.install(json_license_path="license.json")

nlp.start()

class AssertionTests(unittest.TestCase):
    def test_few_shot_assertion_model(self):
        # data = 'Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain'
        # data = """Miss M. is a 67-year-old lady, with past history of COPD and Hypertension, presents with a 3-weeks history of a lump in her right Breast. The lump appeared suddenly, also painful. 5 days ago, another lump appeared in her right axilla. On examination a 2 x 3 cm swelling was seen in the right Breast. It was firm and also non-tender and immobile. There was no discharge. Another 1x1 cm circumferential swelling was found in the right Axilla, which was freely mobile and also tender. Her family history is remarkable for Breast cancer  (mother), cervical cancer (maternal grandmother), heart disease (father), COPD (Brother), dementia (Grandfather), diabetes (Grandfather), and CHF (Grandfather)."""
        # res = nlu.load('en.assert.healthcare', verbose=True).predict(data, metadata=True)  # .predict(data)
        data = (
            """Includes hypertension and chronic obstructive pulmonary disease."""
        )

        res = nlu.load("en.few_assert_shot_classifier", verbose=True).predict(data, metadata=True)
        print(res.columns)
        for c in res:
            print(res[c])
        print(res)


if __name__ == "__main__":
    AssertionTests().test_few_shot_assertion_model()
