class ImageTableCellDetector:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageTableCellDetector
        return ImageTableCellDetector() \
            .setInputCol("image_region") \
            .setAlgoType("morphops") \
            .setOutputCol("ocr_table_cells")
