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
nlp.start(visual=True)

class DeidentificationTests(unittest.TestCase):
    def test_generic_classifier(self):

        res = nlu.load("bert elmo", verbose=True).predict(
            "DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin"
        )

        # elmo_embeddings and bert_embeddings   is what should be passed 2 the feature asselmber/generic classifier

        # res.show()
        # for os_components in res.columns:
        #     print(os_components)
        #     res.select(os_components).show(truncate=False)
        # res = nlu.load('en.extract_relation', verbose=True).predict('The patient got cancer in my foot and damage in his brain')

        for c in res:
            print(c)
            print(res[c])

        # print(res)


if __name__ == "__main__":
    DeidentificationTests().test_entities_config()
