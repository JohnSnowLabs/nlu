class ImageSplitRegions:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageSplitRegions
        return ImageSplitRegions() \
            .setInputCol("ocr_image") \
            .setInputRegionsCol("region") \
            .setOutputCol("image_region")

# .setInputRegionsCol("ocr_table_16969+
#
#
#
# ") \