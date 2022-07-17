import tests.secrets as sct
import unittest
import nlu
nlu.auth(sct.SPARK_NLP_LICENSE,sct.AWS_ACCESS_KEY_ID,sct.AWS_SECRET_ACCESS_KEY,sct.JSL_SECRET, sct.OCR_LICENSE, sct.OCR_SECRET)
# nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)

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
        img_path = 'tests/datasets/ocr/table_pdf_highlightable_text/data.pdf'
        p = nlu.load('pdf2table',verbose=True)
        dfs = p.predict(img_path)
        for df in dfs :
            print(df)

    def test_PPT_table_extraction(self):
        f1 = 'tests/datasets/ocr/table_PPT/54111.ppt'
        f2 ='tests/datasets/ocr/table_PPT/mytable.ppt'
        p = nlu.load('ppt2table',verbose=True)
        dfs = p.predict([f1,f2])
        for df in dfs :
            print(df)

    def test_DOC_table_extraction(self):
        f1 = 'tests/datasets/ocr/docx_with_table/doc2.docx'
        p = nlu.load('doc2table',verbose=True)
        dfs = p.predict([f1])
        for df in dfs :
            print(df)


if __name__ == '__main__':
    unittest.main()

