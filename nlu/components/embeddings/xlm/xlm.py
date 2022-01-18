from sparknlp.annotator import XlmRoBertaEmbeddings

class XLM:
    @staticmethod
    def get_default_model():
        return XlmRoBertaEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("xlm_roberta")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return XlmRoBertaEmbeddings.pretrained(name, language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("xlm_roberta")



