import unittest
import pandas as pd
import nlu
from sparknlp.annotator import BertSentenceEmbeddings
import tests.nlu_hc_tests.secrets as sct

class SentenceResolverTrainingTests(unittest.TestCase):
    def test_sentence_resolver_training(self):
        """When training a chunk resolver, word_embedding are required.
        If none specifeid, the default `glove` word_embeddings will be used
        Alternatively, if a Word Embedding is specified in the load command before the train.chunk_resolver,
        it will be used instead of the default glove
        """
        import pandas as pd
        cols = ["y","_y","text"]
        p='/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tests/datasets/AskAPatient.fold-0.train.txt'
        dataset = pd.read_csv(p,sep="\t",encoding="ISO-8859-1",header=None)
        dataset.columns = cols
        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET

        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)

        trainable_pipe = nlu.load('train.resolve_sentence', verbose=True)
        trainable_pipe.print_info()
        fitted_pipe  = trainable_pipe.fit(dataset)
        res = fitted_pipe.predict(dataset, multithread=False)


        for c in res :
            print(c)
            print(res[c])

        # print(res)


    def test_chunk_resolver_training_custom_embeds(self):
        """When training a chunk resolver, word_embedding are required.
        If none specifeid, the default `glove` word_embeddings will be used
        Alternatively, if a Word Embedding is specified in the load command before the train.chunk_resolver,
        it will be used instead of the default glove
        """
        import pandas as pd
        cols = ["y","_y","text"]
        p='/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tests/datasets/AskAPatient.fold-0.train.txt'
        dataset = pd.read_csv(p,sep="\t",encoding="ISO-8859-1",header=None)
        dataset.columns = cols

        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET

        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)

        # trainable_pipe = nlu.load('glove train.resolve_chunks', verbose=True)
        trainable_pipe = nlu.load('embed_sentence.bert train.resolve_sentence', verbose=True)
        trainable_pipe.print_info()
        fitted_pipe  = trainable_pipe.fit(dataset)
        res = fitted_pipe.predict(dataset, multithread=False)


        for c in res :
            print(c)
            print(res[c])

        # print(res)




if __name__ == '__main__':
    SentenceResolverTrainingTests().test_entities_config()

