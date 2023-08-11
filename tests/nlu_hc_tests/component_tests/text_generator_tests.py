import os
import sys
import unittest

import nlu
import secrets as sct

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

text_generator_spells = [
    'en.generate.biomedical_biogpt_base',
    'en.generate.biogpt_chat_jsl_conversational'
]


class MedicalTextGeneratorTests(unittest.TestCase):
    def test_medical_text_generator(self):
        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )


        for s in text_generator_spells:
            pipe = nlu.load(s)
            # Configure relations to extract
            print("TESTING: ", s)
            df = pipe.predict("Covid 19 is",output_level='chunk')
            print(df.columns)
            for c in df:
                print(c)
                print(df[c])

            print(df)


if __name__ == "__main__":
    MedicalTextGeneratorTests().test_medical_text_generator()
