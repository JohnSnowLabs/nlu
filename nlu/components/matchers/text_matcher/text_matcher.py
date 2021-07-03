import sparknlp
from sparknlp.annotator import TextMatcherModel


class TextMatcher:
    @staticmethod
    def get_default_model():
        return sparknlp.annotator.TextMatcherModel() \
            .setInputCols("sentence") \
            .setOutputCol("matched_entity") \



    @staticmethod
    def get_pretrained_model(name, language):
        return sparknlp.annotator.TextMatcherModel.pretrained(name,language) \
            .setInputCols("sentence") \
            .setOutputCol("matched_entity") \


