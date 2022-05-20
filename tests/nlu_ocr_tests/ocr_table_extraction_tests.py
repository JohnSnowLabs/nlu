import tests.secrets as sct
import unittest
import nlu

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
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)
        # TODO TEST MULTI FILES!!
        img_path = '/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tests/datasets/ocr/table_pdf_highlightable_text/data.pdf'
        p = nlu.load('pdf2table',verbose=True)
        dfs = p.predict(img_path)
        for df in dfs :
            print(df)
        import os
        def write_result(uuid: str, content: str):
            filename = 'test.json'
            dirname = os.path.dirname(filename)
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            with open(filename, "w") as fp:
                fp.write(content)
                fp.flush()
                os.fsync(fp.fileno())
        write_result('lel',dfs.to_json())

    def test_PPT_table_extraction(self):
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)
        f1 = '/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tests/datasets/ocr/table_PPT/54111.ppt'
        f2 ='/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tests/datasets/ocr/table_PPT/mytable.ppt'
        p = nlu.load('ppt2table',verbose=True)
        dfs = p.predict([f1,f2])
        # |PdfToTextTable_06cb624a0f81:Error: Header doesn't contain versioninfo|
        # |PdfToTextTable_06cb624a0f81:Error: Header doesn't contain versioninfo|
        for df in dfs :
            print(df)

    def test_DOC_table_extraction(self):
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)
        f1 = '/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tests/datasets/ocr/table_DOCX/doc2.docx'
        p = nlu.load('doc2table',verbose=True)
        dfs = p.predict([f1])
        for df in dfs :
            print(df)


if __name__ == '__main__':
    unittest.main()

