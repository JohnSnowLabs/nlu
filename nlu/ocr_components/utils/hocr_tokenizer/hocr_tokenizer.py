class HocrTokenizer:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import HocrTokenizer
        return HocrTokenizer() \
            .setInputCol("hocr") \
            .setOutputCol("text_tokenized")
