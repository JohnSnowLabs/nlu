class IMAGE_TABLE_DETECTOR:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageTableDetector
        return ImageTableDetector.pretrained("general_model_table_detection_v2", "en", "clinical/ocr") \
            .setInputCol("ocr_image") \
            .setOutputCol("region")
