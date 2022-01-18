class Pdf2Text:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import PdfToText
        return PdfToText() \
            .setInputCol("content") \
            .setOutputCol("text") \
            .setPageNumCol("pagenum")

