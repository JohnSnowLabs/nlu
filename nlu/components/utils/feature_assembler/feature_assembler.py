"""
The FeaturesAssembler is used to collect features from different columns. 
It can collect features from single value columns (anything which can be cast to a float, if casts fails then the value is set to 0),
 array columns or SparkNLP annotations (if the annotation is an embedding, it takes the embedding, otherwise tries to cast the result field). 
The output of the transformer is a FEATURE_VECTOR annotation (the numeric vector is in the embeddings field).
"""
class SparkNLPFeatureAssembler:
    @staticmethod

    def get_default_model():
        from sparknlp_jsl.base import FeaturesAssembler
        return FeaturesAssembler() \
            .setInputCols(["%%%feature_elements%%%"]) \
            .setOutputCol("feature_vector")
