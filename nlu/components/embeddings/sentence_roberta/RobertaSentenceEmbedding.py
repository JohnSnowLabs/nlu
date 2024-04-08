from sparknlp.annotator import RoBertaSentenceEmbeddings


class RobertaSentence:
    @staticmethod
    def get_default_model():
        return RoBertaSentenceEmbeddings.pretrained() \
            .setInputCols("sentence") \
            .setOutputCol("sentence_embeddings")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return RoBertaSentenceEmbeddings.pretrained(name,language,bucket) \
            .setInputCols('sentence') \
            .setOutputCol("sentence_embeddings")



