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
spark = nlp.start(visual=True)
class DeidentificationTests_OCR(unittest.TestCase):
    def test_deidentification(self):

        res = nlu.load("en.image_deid")
        input_path, output_path = ['download.pdf', 'deid2.pdf'], [
            'F:\\Work\\repos\\nlu\\tests\\datasets\\ocr\\download_deidentified.pdf',
            'F:\\Work\\repos\\nlu\\tests\\datasets\\ocr\\deid\\deid2_deidentified.pdf']

        dfs = res.predict(input_path, output_path=output_path)


if __name__ == "__main__":
    DeidentificationTests_OCR().test_deidentification()
