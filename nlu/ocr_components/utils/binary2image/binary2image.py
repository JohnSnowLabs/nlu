class Binary2Image:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import BinaryToImage
        return BinaryToImage() \
            .setInputCol("content") \
            .setOutputCol("image")
