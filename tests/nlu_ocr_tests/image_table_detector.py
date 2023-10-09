# import tests.secrets as sct
import os
import sys

import unittest
import nlu

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from johnsnowlabs import nlp,visual
nlp.install(visual=True,json_license_path='5.1.1.spark_nlp_for_healthcare (1).json')
nlp.start(visual=True)
# SPARK_NLP_LICENSE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3MDQwNjcyMDAsImlhdCI6MTY3Mjk2MzIwMCwidW5pcXVlX2lkIjoiZGQ0MzE4ZTYtOGRhOS0xMWVkLTgyNjAtY2ViMjJiMTM3OTk4Iiwic2NvcGUiOlsibGVnYWw6aW5mZXJlbmNlIiwibGVnYWw6dHJhaW5pbmciLCJmaW5hbmNlOmluZmVyZW5jZSIsImZpbmFuY2U6dHJhaW5pbmciLCJvY3I6aW5mZXJlbmNlIiwib2NyOnRyYWluaW5nIiwiaGVhbHRoY2FyZTppbmZlcmVuY2UiLCJoZWFsdGhjYXJlOnRyYWluaW5nIl19.Uw5z6ihpLukV9sBVZn4SRZmgshmLaIFHc_KqNGKejS7Yj4b3m0pM7FMRBx2BJ5rzIPQJD0P0Qv-vK42Ze71BS4_TDe0r52UltmxX0K1R4ijUbK3gA0qYJMSRZnFSKIocZ7TRxXcACJeHsqnMkp6um0D7abrdKMSdzEM87TAOX0sO8H29rhW8UKz5eiE3o45hMMcYuxFv5zbJr9X7pxZkbVmI72Mbq8Pq0PXzKIct1S85IhKo22tlhgGeo_CLGZkDsM9735QiBTqZ8olX5sFpqTy4cDMuoX5odR8VBumf37w80NYEIZlt_vOWaXEgWvYGDhjYxJ-YbUv0bT9kQ4TmHA"
# AWS_ACCESS_KEY_ID = "AKIASRWSDKBGFCFZ6P4H"
# AWS_SECRET_ACCESS_KEY = "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"
# JSL_SECRET = "5.0.2-2edb671dfc23389dc2b428f9c49244415b340f34"
# OCR_LICENSE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3MDQwNjcyMDAsImlhdCI6MTY3Mjk2MzIwMCwidW5pcXVlX2lkIjoiZGQ0MzE4ZTYtOGRhOS0xMWVkLTgyNjAtY2ViMjJiMTM3OTk4Iiwic2NvcGUiOlsibGVnYWw6aW5mZXJlbmNlIiwibGVnYWw6dHJhaW5pbmciLCJmaW5hbmNlOmluZmVyZW5jZSIsImZpbmFuY2U6dHJhaW5pbmciLCJvY3I6aW5mZXJlbmNlIiwib2NyOnRyYWluaW5nIiwiaGVhbHRoY2FyZTppbmZlcmVuY2UiLCJoZWFsdGhjYXJlOnRyYWluaW5nIl19.Uw5z6ihpLukV9sBVZn4SRZmgshmLaIFHc_KqNGKejS7Yj4b3m0pM7FMRBx2BJ5rzIPQJD0P0Qv-vK42Ze71BS4_TDe0r52UltmxX0K1R4ijUbK3gA0qYJMSRZnFSKIocZ7TRxXcACJeHsqnMkp6um0D7abrdKMSdzEM87TAOX0sO8H29rhW8UKz5eiE3o45hMMcYuxFv5zbJr9X7pxZkbVmI72Mbq8Pq0PXzKIct1S85IhKo22tlhgGeo_CLGZkDsM9735QiBTqZ8olX5sFpqTy4cDMuoX5odR8VBumf37w80NYEIZlt_vOWaXEgWvYGDhjYxJ-YbUv0bT9kQ4TmHA"
# OCR_SECRET = "5.0.0-9fd5dda7491d999a05c9bdac4b92a046694e8116"
#
# nlu.auth(SPARK_NLP_LICENSE ,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)

class OCRTests(unittest.TestCase):
    # def test_ocr(self):
    #
    #     img_path = 'table.png'
    #
    #     p = nlu.load('image_table_detector',verbose=True)
    #     dfs = p.predict(img_path)
    #     for df in dfs:
    #         print(df)
    # def test_ocr(self):
    #     img_path = 'test2.jpg'
    #     p = nlu.load('img2text',verbose=True)
    #     dfs = p.predict(img_path)
    #     for df in dfs :
    #         print(df)
    def test_DOC_table_extraction(self):
        # f1 = 'doc2.docx'
        img_path = 'data.pdf'
        p = nlu.load('image_table_detector image_table_cell_detector image_table_cell2text_table',verbose=True)
        # p['image_split_regions'].setInputRegionsCol('ocr_table_1')
        # p['image_table_cell2text_table'].setCellsCol('ocr_table_cells')
        #p['image_table_cell2text_table'].setCellsCol('ocr_table_3')
        dfs = p.predict('table.png')
        for df in dfs:
            print(df)

if __name__ == "__main__":
    OCRTests().test_DOC_table_extraction()


