# import tests.secrets as sct
import os
import sys
from johnsnowlabs import nlp
import unittest
import nlu
from johnsnowlabs import nlp,visual
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# nlp.install(visual=True)
# nlp.start(visual=True)
SPARK_NLP_LICENSE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3MDQwNjcyMDAsImlhdCI6MTY3Mjk2MzIwMCwidW5pcXVlX2lkIjoiZGQ0MzE4ZTYtOGRhOS0xMWVkLTgyNjAtY2ViMjJiMTM3OTk4Iiwic2NvcGUiOlsibGVnYWw6aW5mZXJlbmNlIiwibGVnYWw6dHJhaW5pbmciLCJmaW5hbmNlOmluZmVyZW5jZSIsImZpbmFuY2U6dHJhaW5pbmciLCJvY3I6aW5mZXJlbmNlIiwib2NyOnRyYWluaW5nIiwiaGVhbHRoY2FyZTppbmZlcmVuY2UiLCJoZWFsdGhjYXJlOnRyYWluaW5nIl19.Uw5z6ihpLukV9sBVZn4SRZmgshmLaIFHc_KqNGKejS7Yj4b3m0pM7FMRBx2BJ5rzIPQJD0P0Qv-vK42Ze71BS4_TDe0r52UltmxX0K1R4ijUbK3gA0qYJMSRZnFSKIocZ7TRxXcACJeHsqnMkp6um0D7abrdKMSdzEM87TAOX0sO8H29rhW8UKz5eiE3o45hMMcYuxFv5zbJr9X7pxZkbVmI72Mbq8Pq0PXzKIct1S85IhKo22tlhgGeo_CLGZkDsM9735QiBTqZ8olX5sFpqTy4cDMuoX5odR8VBumf37w80NYEIZlt_vOWaXEgWvYGDhjYxJ-YbUv0bT9kQ4TmHA"
AWS_ACCESS_KEY_ID = "AKIASRWSDKBGFCFZ6P4H"
AWS_SECRET_ACCESS_KEY = "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"
JSL_SECRET = "5.0.2-2edb671dfc23389dc2b428f9c49244415b340f34"
OCR_LICENSE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3MDQwNjcyMDAsImlhdCI6MTY3Mjk2MzIwMCwidW5pcXVlX2lkIjoiZGQ0MzE4ZTYtOGRhOS0xMWVkLTgyNjAtY2ViMjJiMTM3OTk4Iiwic2NvcGUiOlsibGVnYWw6aW5mZXJlbmNlIiwibGVnYWw6dHJhaW5pbmciLCJmaW5hbmNlOmluZmVyZW5jZSIsImZpbmFuY2U6dHJhaW5pbmciLCJvY3I6aW5mZXJlbmNlIiwib2NyOnRyYWluaW5nIiwiaGVhbHRoY2FyZTppbmZlcmVuY2UiLCJoZWFsdGhjYXJlOnRyYWluaW5nIl19.Uw5z6ihpLukV9sBVZn4SRZmgshmLaIFHc_KqNGKejS7Yj4b3m0pM7FMRBx2BJ5rzIPQJD0P0Qv-vK42Ze71BS4_TDe0r52UltmxX0K1R4ijUbK3gA0qYJMSRZnFSKIocZ7TRxXcACJeHsqnMkp6um0D7abrdKMSdzEM87TAOX0sO8H29rhW8UKz5eiE3o45hMMcYuxFv5zbJr9X7pxZkbVmI72Mbq8Pq0PXzKIct1S85IhKo22tlhgGeo_CLGZkDsM9735QiBTqZ8olX5sFpqTy4cDMuoX5odR8VBumf37w80NYEIZlt_vOWaXEgWvYGDhjYxJ-YbUv0bT9kQ4TmHA"
OCR_SECRET = "5.0.0-9fd5dda7491d999a05c9bdac4b92a046694e8116"
nlp.start(SPARK_NLP_LICENSE ,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)
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
        binary_to_image = visual.BinaryToImage()
        binary_to_image.setOutputCol("image")
        binary_to_image.setImageType(visual.ImageType.TYPE_3BYTE_BGR)

        # Detect tables on the page using pretrained model
        # It can be finetuned for have more accurate results for more specific documents
        table_detector = visual.ImageTableDetector.pretrained("general_model_table_detection_v2", "en", "clinical/ocr")
        table_detector.setInputCol("image")
        table_detector.setOutputCol("region")

        # Draw detected region's with table to the page
        draw_regions = visual.ImageDrawRegions()
        draw_regions.setInputCol("image")
        draw_regions.setInputRegionsCol("region")
        draw_regions.setOutputCol("image_with_regions")
        draw_regions.setRectColor(visual.Color.red)

        # Extract table regions to separate images
        splitter = visual.ImageSplitRegions()
        splitter.setInputCol("image")
        splitter.setInputRegionsCol("region")
        splitter.setOutputCol("table_image")
        splitter.setDropCols("image")

        # Detect cells on the table image
        cell_detector = visual.ImageTableCellDetector()
        cell_detector.setInputCol("table_image")
        cell_detector.setOutputCol("cells")
        cell_detector.setAlgoType("morphops")
        cell_detector.setDrawDetectedLines(True)

        # Extract text from the detected cells
        table_recognition = visual.ImageCellsToTextTable()
        table_recognition.setInputCol("table_image")
        table_recognition.setCellsCol('cells')
        table_recognition.setMargin(3)
        table_recognition.setStrip(True)
        table_recognition.setOutputCol('table')

        # Erase detected table regions
        fill_regions = visual.ImageDrawRegions()
        fill_regions.setInputCol("image")
        fill_regions.setInputRegionsCol("region")
        fill_regions.setOutputCol("image_1")
        fill_regions.setRectColor(visual.Color.white)
        fill_regions.setFilledRect(True)

        # OCR
        ocr = visual.ImageToText()
        ocr.setInputCol("image_1")
        ocr.setOutputCol("text")
        ocr.setOcrParams(["preserve_interword_spaces=1", ])
        ocr.setKeepLayout(True)
        ocr.setOutputSpaceCharacterWidth(8)

        pipeline_table = PipelineModel(stages=[
            binary_to_image,
            table_detector,
            draw_regions,
            fill_regions,
            splitter,
            cell_detector,
            table_recognition,
            ocr
        ])

        tables_results = pipeline_table.transform(df).cache()

if __name__ == "__main__":
    OCRTests().test_DOC_table_extraction()


