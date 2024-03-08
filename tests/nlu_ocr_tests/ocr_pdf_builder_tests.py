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

    def test_text_to_pdf(self):
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)
        # text that we generate PDF to has to come from an image struct!
        # We need convert text to img struct!

        p = nlu.load('ppt2table',verbose=True)
        dfs = p.predict([f1,f2])
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

