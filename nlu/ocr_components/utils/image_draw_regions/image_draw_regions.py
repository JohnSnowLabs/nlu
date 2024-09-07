class ImageDrawRegions:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageDrawRegions
        return ImageDrawRegions() \
            .setInputCol("ocr_image") \
            .setInputRegionsCol("ocr_positions") \
            .setOutputCol("image_with_regions") \
            .setFilledRect(True)

# .setInputRegionsCol("ocr_table_16969+
#
#
#
# ") \