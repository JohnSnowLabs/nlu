from sparknlp.annotator import *


class LabeledDependencyParser:
    @staticmethod
    def get_default_model():
        return TypedDependencyParserModel.pretrained() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labled_dependency")

    @staticmethod
    def get_pretrained_model(name, language):
        return TypedDependencyParserModel.pretrained(name,language) \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labled_dependency")


    @staticmethod
    def get_default_trainable_model():
        return TypedDependencyParserApproach() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labdep") \
            .setConll2009("file://conll2009/eng.train") \
            .setNumberOfIterations(10)
