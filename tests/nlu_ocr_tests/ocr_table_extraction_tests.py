import sparknlp.annotator
from sparkocr.transformers import *
import tests.secrets as sct
import unittest
from nlu import *
import sparknlp_jsl.annotator

SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
JSL_SECRET            = sct.JSL_SECRET
OCR_SECRET            = sct.OCR_SECRET
OCR_LICENSE           = sct.OCR_LICENSE
# nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)

class OcrTest(unittest.TestCase):
    def test_ocr_and_hc_auth_and_install(self):
        ocr_secret_json = '/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tests/spark_ocr.json'
        hc_secret_json = '/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tests/spark_nlp_for_healthcare.json'
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)


    def test_table_extraction(self):
        """:cvar
        1. PdfToTextAble
        2. DocToTextTable
        3. PptToTextTable

        4.1 ImageTableDetector --> Find Locations of Tables
        4.2 Image TableCell Detector ---> FInd Location of CELLS on the table
        4.3 ImageCellsToTextTable ----> Find TEXT inside of the Cells on the table

        """
        """:cvar
        Whats the difference between DocToTextTable transformer and 
        using ImageTableDetector + ImageTableCellDetector + ImageCellsToTextTable
        The first is pragamtic and the second one is DL based?
        When to use which annotator?
        
        
        ---> for NON SELECTABLE TEXT ImageTableDetector + ImageTableCellDetector + ImageCellsToTextTable
        ---> For text whci his selectable DocToTextTable3.
        """

        nlu.load('table_from_pdf ')

if __name__ == '__main__':
    unittest.main()

