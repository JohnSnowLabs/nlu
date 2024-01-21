import unittest

from nlu import *


class TestBGESentenceEmbeddings(unittest.TestCase):
    def test_bge_embeds(self):
        pipe = nlu.load("en.embed_sentence.bge_small", verbose=True)
        res = pipe.predict(
            "query: how much protein should a female eat",
            output_level="document"
        )
        for c in res:
            print(res[c])


if __name__ == "__main__":
    unittest.main()
