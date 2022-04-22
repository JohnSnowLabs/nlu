class Doc2TextTable:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import DocToTextTable
        return DocToTextTable() \
            .setInputCol("content") \
            .setOutputCol("table")
