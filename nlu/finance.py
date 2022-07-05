#
# # Thin wrapper around sparknlp_jsl classes
# --> Spark NLP annotator Objects
# from johnsnowlabs import Finance, Legal, Medical
# johnsnowlabs.start()
#
# # Under the hood using MedcicalNer/EnterpriseNer Scala class
# fin_ner = Finance.Ner.pretrained().setInputCols() .....
#
# NerDl
# MedicalNerDl
# FinanceNerDl
# # Put in NLU?
# model = MedicalnerDl.pretrained('whaletver')
#
# nlu_pipe = nlu.auto_complete(model)
# nlu_pipe.vanila_pipeline.stages # <----
from sparknlp.common import AnnotatorModel
import sparknlp.annotator
from sparknlp.annotator import *
from sparknlp.internal import ExtendedJavaWrapper

class _FinanceNerModelLoader(ExtendedJavaWrapper):
    def __init__(self, ner_model_path, path, jspark):
        super(_FinanceNerModelLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.ner.MedicalNerModel.loadSavedModel", ner_model_path, path, jspark)

class FinanceNer(AnnotatorModel, HasStorageRef, HasBatchedAnnotate):
    name = "MedicalNerModel"
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.MedicalNerModel", java_model=None):
        super(FinanceNer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            includeConfidence=False,
            includeAllConfidenceScores=False,
            batchSize=8,
            inferenceBatchSize=1
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)
    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "whether to include confidence scores in annotation metadata",
                              TypeConverters.toBoolean)
    includeAllConfidenceScores = Param(Params._dummy(), "includeAllConfidenceScores",
                                       "whether to include all confidence scores in annotation metadata or just the score of the predicted tag",
                                       TypeConverters.toBoolean)
    inferenceBatchSize = Param(Params._dummy(), "inferenceBatchSize",
                               "number of sentences to process in a single batch during inference",
                               TypeConverters.toInt)
    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this MedicalNerModel",
                    TypeConverters.toListString)

    trainingClassDistribution = Param(Params._dummy(),
                                      "trainingClassDistribution",
                                      "class counts for each of the classes during training",
                                      typeConverter=TypeConverters.identity)

    labelCasing = Param(Params._dummy(), "labelCasing",
                        "Setting all labels of the NER models upper/lower case. values upper|lower",
                        TypeConverters.toString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setIncludeConfidence(self, value):
        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        value : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=value)

    def setIncludeConfidence(self, value):
        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        value : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=value)

    def setInferenceBatchSize(self, value):
        """Sets number of sentences to process in a single batch during inference

        Parameters
        ----------
        value : int
           number of sentences to process in a single batch during inference
        """
        return self._set(inferenceBatchSize=value)

    def setLabelCasing(self, value):
        """Setting all labels of the NER models upper/lower case. values upper|lower

        Parameters
        ----------
        value : str
           Setting all labels of the NER models upper/lower case. values upper|lower
        """
        return self._set(labelCasing=value)

    def getTrainingClassDistribution(self):
        return self._call_java('getTrainingClassDistributionJava')

    @staticmethod
    def pretrained(name="ner_dl", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(FinanceNer, name, lang, remote_loc)

    @staticmethod
    def loadSavedModel(ner_model_path, folder, spark_session):
        jModel = _FinanceNerModelLoader(ner_model_path, folder, spark_session._jsparkSession)._java_obj
        return FinanceNer(java_model=jModel)

    @staticmethod
    def pretrained(name="ner_clinical", lang="en", remote_loc="clinical/models"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(FinanceNer, name, lang, remote_loc,
                                                j_dwn='InternalsPythonResourceDownloader')


#from nlu import finance
# finance.FinanceNER