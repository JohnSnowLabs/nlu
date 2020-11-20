import sparknlp
class TextMatcher:
    @staticmethod
    def get_default_model():
        return   sparknlp.annotator.TextMatcherModel() \
            .setInputCols("document", "token") \
            .setOutputCol("entity") \


    @staticmethod
    def get_pretrained_model(name, language):
        return   sparknlp.annotator.TextMatcherModel.pretrained(name,language) \
            .setInputCols("document") \
            .setOutputCol("entity") \


