# import tests.secrets as sct

import os
import sys

# sys.path.append(os.getcwd())
import unittest
import nlu

os.environ["PYTHONPATH"] = "F:/Work/repos/nlu_new/nlu"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from johnsnowlabs import nlp, visual

# nlp.install(json_license_path="license.json")

nlp.start()

class EmbeddingTests(unittest.TestCase):
    def test_mxbai_embeddings_model(self):

        res = nlu.load("en.embed.mxbai").predict('This is an example sentence', output_level='document')
        print(res)


if __name__ == "__main__":
    EmbeddingTests().test_mxbai_embeddings_model()
