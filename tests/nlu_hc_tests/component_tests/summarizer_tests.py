import os
import sys
import unittest

import nlu
import tests.secrets as sct

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

summarizer_spells = [
    'en.summarize.clinical_jsl',
    'en.summarize.clinical_jsl_augmented',
    'en.summarize.biomedical_pubmed',
    'en.summarize.generic_jsl',
    'en.summarize.clinical_questions',
    'en.summarize.radiology',
    'en.summarize.clinical_guidelines_large',
    'en.summarize.clinical_laymen',
]


class MedicalSummarizerTests(unittest.TestCase):
    def test_medical_summarizer(self):
        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )
        # b = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')


        for s in summarizer_spells:
            pipe = nlu.load(s)
            # Configure relations to extract
            print("TESTING: ", s)
            df = pipe.predict("Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer.")
            print(df.columns)
            for c in df:
                print(c)
                print(df[c])

            print(df)


if __name__ == "__main__":
    MedicalSummarizerTests().test_medical_summarizer()
