import os
import sys
import unittest
sys.path.append(os.getcwd())

import nlu

os.environ["PYTHONPATH"] = "F:/Work/repos/nlu"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from johnsnowlabs import nlp, visual
# nlp.settings.enforce_versions=False
# nlp.install(json_license_path='license.json',visual=True)
spark = nlp.start()
class DeidentificationTests(unittest.TestCase):
    def test_deidentification(self):
        # b = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')
        # m = RelationExtractionModel().pretrained("posology_re")
        #
        # res = nlu.load('en.ner.deid.augmented  en.de_identify', verbose=True).predict('DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin', return_spark_df=True)

        res = nlu.load("en.de_identify").predict(
            "DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin",
            drop_irrelevant_cols=False,
            metadata=True,
        )
        # res = nlu.load('zh.segment_words pos', verbose=True)#.predict('DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin', return_spark_df=True)

        for c in res:
            print(c)
            print(res[c])

        # print(res)


if __name__ == "__main__":
    DeidentificationTests().test_deidentification()
