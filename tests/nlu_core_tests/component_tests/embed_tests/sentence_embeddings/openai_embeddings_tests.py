import unittest
from nlu import *
import os


class TestOpenAIEmbeddings(unittest.TestCase):
    def test_openAI_embeds(self):

        pipe = nlu.load("openai.embeddings")

        pipe['openai_embeddings'].setModel('text-embedding-ada-002')

        res = pipe.predict(["The food was delicious and the waiter...","canine companions say"], output_level='document')

        for c in res:
            print(res[c])

if __name__ == "__main__":
    unittest.main()
