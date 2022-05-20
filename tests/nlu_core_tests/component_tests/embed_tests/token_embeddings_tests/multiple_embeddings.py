import unittest

from nlu import *


class TestMultipleEmbeddings(unittest.TestCase):
    def test_multiple_embeddings(self):
        df = nlu.load(
            "bert en.embed.bert.small_L8_512 en.embed.bert.small_L8_512 en.embed.bert.small_L8_128  electra en.embed.bert.small_L10_128 en.embed.bert.small_L4_128",
            verbose=True,
        ).predict("Am I the muppet or are you the muppet?")
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
