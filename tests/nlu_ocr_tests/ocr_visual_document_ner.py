import os
import sys

sys.path.append(os.getcwd())
import unittest
import nlu

os.environ["PYTHONPATH"] = "F:/Work/repos/nlu"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from johnsnowlabs import nlp, visual

# nlp.install(json_license_path='license.json',visual=True)
nlp.start(visual=True)

class OcrTest(unittest.TestCase):

    def test_classify_document(self):
        # nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)
        # text that we generate PDF to has to come from an image struct!
        # We need convert text to img struct!
        p = nlu.load('en.lilt_roberta_funds.v1').predict('ocr_ner.png',output_level='chunk')
        for i,j in p.iterrows():
            print(i,'---->',j)

if __name__ == '__main__':
    unittest.main()