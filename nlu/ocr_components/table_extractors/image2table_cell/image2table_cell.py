class ImageTableCellDetector:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageTableCellDetector
        return ImageTableCellDetector() \
            .setInputCol("table_image") \
            .setAlgoType("morphops") \
            .setOutputCol("cells")
