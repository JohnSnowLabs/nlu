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

    def test_PDF_table_extraction(self):
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
        # img_path = 'tests/datasets/ocr/table_pdf_highlightable_text/data.pdf'
        # p = nlu.load('pdf2table',verbose=True)
        # dfs = p.predict(img_path)
        # for df in dfs :
        #     print(df)

    def test_PPT_table_extraction(self):
        f1 = '54111.ppt'
        f2 ='tests/datasets/ocr/table_PPT/mytable.ppt'
        p = nlu.load('ppt2table',verbose=True)
        dfs = p.predict([f1    ])
        for df in dfs :
            print(df)
    #
    # def test_DOC_table_extraction(self):
    #     f1 = 'tests/datasets/ocr/docx_with_table/doc2.docx'
    #     p = nlu.load('doc2table',verbose=True)
    #     dfs = p.predict([f1])
    #     for df in dfs :
    #         print(df)


if __name__ == '__main__':
    unittest.main()

