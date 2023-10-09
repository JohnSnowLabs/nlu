class ImageSplitRegions:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageSplitRegions
        return ImageSplitRegions() \
            .setInputCol("ocr_image") \
            .setInputRegionsCol("ocr_table_1") \
            .setOutputCol("table_image")

# .setInputRegionsCol("ocr_table_1") \