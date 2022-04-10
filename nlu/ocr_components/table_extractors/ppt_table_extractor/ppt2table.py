class PPT2TextTable:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import PptToTextTable
        return PptToTextTable() \
            .setInputCol("content") \
            .setOutputCol("table")
