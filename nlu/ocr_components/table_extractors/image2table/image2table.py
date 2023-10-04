class IMAGE_TABLE_DETECTOR:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageTableDetector
        return ImageTableDetector() \
            .setInputCol("ocr_image") \
            .setOutputCol("region")
