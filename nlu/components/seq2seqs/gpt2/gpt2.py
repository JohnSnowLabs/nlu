from sparknlp.annotator import GPT2Transformer

class GPT2:
    @staticmethod
    def get_default_model():
        return GPT2Transformer.pretrained() \
            .setInputCols("document") \
            .setOutputCol("gpt2")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return GPT2Transformer.pretrained(name, language) \
            .setInputCols("document") \
            .setOutputCol("gpt2")



