class PDF2TextTable:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import PdfToTextTable
        return PdfToTextTable() \
            .setInputCol("content") \
            .setOutputCol("table")
