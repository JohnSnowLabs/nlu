from sparknlp.annotator import *


class UnlabeledDependencyParser:
    @staticmethod
    def get_default_model():
        return DependencyParserModel.pretrained() \
            .setInputCols(["document", "pos", "token"]) \
            .setOutputCol("dependency")

    @staticmethod
    def get_pretrained_model(name, language):
        return DependencyParserModel.pretrained(name,language) \
            .setInputCols(["document", "pos", "token"]) \
            .setOutputCol("dependency")


    @staticmethod
    def get_default_trainable_model():
        return DependencyParserApproach() \
            .setInputCols(["document", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setDependencyTreeBank("file://parser/dependency_treebank") \
            .setNumberOfIterations(10)
