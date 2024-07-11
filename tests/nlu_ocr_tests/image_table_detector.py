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

# nlp.install(visual=True,json_license_path="license.json")
nlp.start(visual=True)

class OCRTests(unittest.TestCase):

    def test_DOC_table_extraction(self):


        p = nlu.load('image_table_cell2text_table', verbose=True)

        images = ['form.png']
        for path in images:
            dfs = p.predict(path)
            print(dfs)
            print("***" * 30)


if __name__ == "__main__":
    OCRTests().test_DOC_table_extraction()


