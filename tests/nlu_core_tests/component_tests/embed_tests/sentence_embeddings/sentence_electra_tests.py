import unittest

from nlu import *


class TestElectraSentenceEmbeddings(unittest.TestCase):
    def test_electra_sentence_embeds(self):
        res = nlu.load("embed_sentence.electra", verbose=True).predict(
            "Am I the muppet or are you the muppet?"
        )
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])
        for c in res:
            print(res[c])

        res = nlu.load("en.embed_sentence.electra", verbose=True).predict(
            "Am I the muppet or are you the muppet?"
        )
        # df = nlu.load('en.classify.sarcasm',verbose=True).predict(sarcasm_df['text'])
        for c in res:
            print(res[c])

        # df = nlu.load('en.embed.bert.small_L4_128', verbose=True).predict("No you are the muppet!")
        # print(df.columns)
        # print(df)
        # print(df['bert_embeddings'])


if __name__ == "__main__":
    unittest.main()
