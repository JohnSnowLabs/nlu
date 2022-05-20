import unittest

import pandas as pd

import nlu
import tests.secrets as sct


class ChunkResolverTrainingTests(unittest.TestCase):
    def test_chunk_resolver_training(self):
        """When training a chunk resolver, word_embedding are required.
        If none specifeid, the default `glove` word_embeddings will be used
        Alternatively, if a Word Embedding is specified in the load command before the train.chunk_resolver,
        it will be used instead of the default glove
        """
        dataset = pd.DataFrame(
            {
                "text": [
                    "super sleepy",
                    "bleeding from ears",
                    "bleeding from nose",
                    "bleeding from mouth",
                ],
                "_extra_info": [
                    "bad disease",
                    "bad disease!",
                    "very bad",
                    " super bad  ",
                ],
                # 'y': [1,33,44,66]
                # 'label': ['lol','kek','lol','kek']
                "label": ["lol", "kek", "lol", "kek"],
            }
        )

        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET

        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )

        trainable_pipe = nlu.load("train.assert", verbose=True)
        trainable_pipe = nlu.load("train.assert_dl", verbose=True)
        trainable_pipe.print_info()
        fitted_pipe = trainable_pipe.fit(dataset)
        res = fitted_pipe.predict(dataset, multithread=False)

        for c in res:
            print(c)
            print(res[c])

        # print(res)

    def test_chunk_resolver_training_custom_embeds(self):
        """When training a chunk resolver, word_embedding are required.
        If none specifeid, the default `glove` word_embeddings will be used
        Alternatively, if a Word Embedding is specified in the load command before the train.chunk_resolver,
        it will be used instead of the default glove
        """
        dataset = pd.DataFrame(
            {
                "text": [
                    "super sleepy",
                    "bleeding from ears",
                    "bleeding from nose",
                    "bleeding from mouth",
                ],
                "_extra_info": [
                    "bad disease",
                    "bad disease!",
                    "very bad",
                    " super bad  ",
                ],
                # 'y': [1,33,44,66]
                # 'label': ['lol','kek','lol','kek']
                "label": ["lol", "kek", "lol", "kek"],
            }
        )

        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET

        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )

        # trainable_pipe = nlu.load('glove train.resolve_chunks', verbose=True)
        trainable_pipe = nlu.load(
            "en.embed.glove.biovec train.resolve_chunks", verbose=True
        )
        # TODO bert/elmo give wierd storage ref errors...
        # TODO WRITE ISSUE IN HEALTHCARE LIB ABOUT THIS!!!
        # ONLY GLOVE WORKS!!
        # trainable_pipe = nlu.load('bert train.resolve_chunks', verbose=True)
        trainable_pipe.print_info()
        fitted_pipe = trainable_pipe.fit(dataset)
        res = fitted_pipe.predict(dataset, multithread=False)

        for c in res:
            print(c)
            print(res[c])

        # print(res)


if __name__ == "__main__":
    ChunkResolverTrainingTests().test_entities_config()
