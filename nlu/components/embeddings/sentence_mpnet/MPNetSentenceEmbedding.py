# from sparknlp.annotator import MPNetEmbeddings
#
#
# class MPNetSentence:
#     @staticmethod
#     def get_default_model():
#         return MPNetEmbeddings.pretrained() \
#             .setInputCols(["documents"]) \
#             .setOutputCol("mpnet_embeddings")
#
#     @staticmethod
#     def get_pretrained_model(name, language, bucket=None):
#         return MPNetEmbeddings.pretrained(name,language,bucket) \
#             .setInputCols(["documents"]) \
#             .setOutputCol("mpnet_embeddings")
#
#
#
