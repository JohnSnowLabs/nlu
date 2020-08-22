import nlu.pipe_components
import sparknlp
from sparknlp.base import *

class SparkNlpDocumentAssembler:
    @staticmethod
    def get_default_model():
        # raw_text_col =
        return DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document") \
            .setCleanupMode("shrink")


