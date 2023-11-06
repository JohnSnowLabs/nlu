import unittest

from nlu import *


class TestMPNetSentenceEmbeddings(unittest.TestCase):
    def test_mpnet_embeds(self):
        res = nlu.load('en.embed_sentence.mpnet.all_mpnet_base_v2').predict('This is an example sentence',
                                                                            output_level='document')
        for c in res:
            print(res[c])

        res = nlu.load('en.embed_sentence.mpnet.all_mpnet_base_questions_clustering_english').predict(
            "Each sentence is converted",
            output_level='document')

        for c in res:
            print(res[c])


if __name__ == "__main__":
    unittest.main()