# import tests.secrets as sct
import os
import sys

import unittest
from pyspark.ml import PipelineModel
import nlu

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from johnsnowlabs import nlp,visual
nlp.install(visual=True, json_license_path='5.1.1.spark_nlp_for_healthcare (1)')
nlp.start(visual=True)

class OCRTests(unittest.TestCase):
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
        data = 'table.png'

        tables_results = pipeline_table.transform(df).cache()

if __name__ == "__main__":
    OCRTests().test_DOC_table_extraction()


