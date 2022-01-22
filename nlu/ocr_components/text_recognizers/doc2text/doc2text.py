class Doc2Text:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import DocToText
        return DocToText() \
            .setInputCol("content") \
            .setOutputCol("text")
