import unittest

import tests.secrets as sct
from nlu import *


class OcrTest(unittest.TestCase):
    def test_ocr_and_hc_auth_and_install(self):
        nlu.auth(
            sct.SPARK_NLP_LICENSE,
            sct.AWS_ACCESS_KEY_ID,
            sct.AWS_SECRET_ACCESS_KEY,
            sct.JSL_SECRET,
            sct.OCR_LICENSE,
            sct.OCR_SECRET,
        )

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

        nlu.load("table_from_pdf")


if __name__ == "__main__":
    unittest.main()
