from sparknlp.annotator import *

class ContextSpellChecker:
    @staticmethod
    def get_default_model():
        return ContextSpellCheckerModel.pretrained() \
            .setInputCols(["token"]) \
            .setOutputCol("spell")
    @staticmethod
    def get_pretrained_model(name, language):
        return ContextSpellCheckerModel.pretrained(name, language) \
            .setInputCols(["token"]) \
            .setOutputCol("spell")



    @staticmethod
    def get_default_trainable_model():
        return ContextSpellCheckerApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("spell")
