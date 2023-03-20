import unittest

from johnsnowlabs import nlp


import nlu
import tests.secrets as sct


class ZeroShotNerTests(unittest.TestCase):
    def test_zero_shot_relation_model(self):
        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )

        pipe = nlu.load('en.zero_shot.ner_roberta')
        print(pipe)
        pipe['zero_shot_ner'].setEntityDefinitions(
            {
                "PROBLEM": [
                    "What is the disease?",
                    "What is his symptom?",
                    "What is her disease?",
                    "What is his disease?",
                    "What is the problem?",
                    "What does a patient suffer",
                    "What was the reason that the patient is admitted to the clinic?",
                ],
                "DRUG": [
                    "Which drug?",
                    "Which is the drug?",
                    "What is the drug?",
                    "Which drug does he use?",
                    "Which drug does she use?",
                    "Which drug do I use?",
                    "Which drug is prescribed for a symptom?",
                ],
                "ADMISSION_DATE": ["When did patient admitted to a clinic?"],
                "PATIENT_AGE": [
                    "How old is the patient?",
                    "What is the gae of the patient?",
                ],
            }
        )

        df = pipe.predict(
            [
                "The doctor pescribed Majezik for my severe headache.",
                "The patient was admitted to the hospital for his colon cancer.",
                "27 years old patient was admitted to clinic on Sep 1st by Dr. X for a right-sided pleural effusion for thoracentesis.",
            ]
        )
        # Configure relationsz to extract
        print(df.columns)
        for c in df:
            print(c)
            print(df[c])

        print(df)


if __name__ == "__main__":
    ZeroShotNerTests().test_entities_config()
