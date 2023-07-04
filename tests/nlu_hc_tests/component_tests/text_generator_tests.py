import os
import sys
import unittest

import nlu
# import secrets as sct

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

text_generator_spells = [
    'en.text_generator.biomedical_biogpt_base'
    # 'en.summarizer.clinical_jsl_augmented',
    # 'en.summarizer.biomedical_pubmed',
    # 'en.summarizer.generic_jsl',
    # 'en.summarizer.clinical_questions',
    # 'en.summarizer.radiology',
    # 'en.summarizer.clinical_guidelines_large',
    # 'en.summarizer.clinical_laymen',
]


class MedicalTextGeneratorTests(unittest.TestCase):
    def test_medical_text_generator(self):
        SPARK_NLP_LICENSE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3MDQwNjcyMDAsImlhdCI6MTY3Mjk2MzIwMCwidW5pcXVlX2lkIjoiZGQ0MzE4ZTYtOGRhOS0xMWVkLTgyNjAtY2ViMjJiMTM3OTk4Iiwic2NvcGUiOlsibGVnYWw6aW5mZXJlbmNlIiwibGVnYWw6dHJhaW5pbmciLCJmaW5hbmNlOmluZmVyZW5jZSIsImZpbmFuY2U6dHJhaW5pbmciLCJvY3I6aW5mZXJlbmNlIiwib2NyOnRyYWluaW5nIiwiaGVhbHRoY2FyZTppbmZlcmVuY2UiLCJoZWFsdGhjYXJlOnRyYWluaW5nIl19.Uw5z6ihpLukV9sBVZn4SRZmgshmLaIFHc_KqNGKejS7Yj4b3m0pM7FMRBx2BJ5rzIPQJD0P0Qv-vK42Ze71BS4_TDe0r52UltmxX0K1R4ijUbK3gA0qYJMSRZnFSKIocZ7TRxXcACJeHsqnMkp6um0D7abrdKMSdzEM87TAOX0sO8H29rhW8UKz5eiE3o45hMMcYuxFv5zbJr9X7pxZkbVmI72Mbq8Pq0PXzKIct1S85IhKo22tlhgGeo_CLGZkDsM9735QiBTqZ8olX5sFpqTy4cDMuoX5odR8VBumf37w80NYEIZlt_vOWaXEgWvYGDhjYxJ-YbUv0bT9kQ4TmHA"
        AWS_ACCESS_KEY_ID = "AKIASRWSDKBGLU62MMFE"
        AWS_SECRET_ACCESS_KEY = "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"
        JSL_SECRET = "4.4.3-d6a8f771226d032db7dccc1eae0635d5fe68d696"
        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )
        # b = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')


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
