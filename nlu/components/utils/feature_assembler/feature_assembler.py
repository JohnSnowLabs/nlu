import nlu.pipe_components
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.base import FeaturesAssembler
class SparkNLPFeatureAssembler:
    @staticmethod
    def get_default_model():
        return FeaturesAssembler() \
            .setInputCols(["sentence", "token", "ner"]) \
            .setOutputCol("feature_vector") \
