from sparknlp.annotator import XlmRoBertaSentenceEmbeddings

class Sentence_XLM:
    @staticmethod
    def get_default_model():
        return XlmRoBertaSentenceEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("sentence_xlm_roberta")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return XlmRoBertaSentenceEmbeddings.pretrained(name, language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("sentence_xlm_roberta")



