import unittest

from nlu import *


class TestxlmEmbeddings(unittest.TestCase):
    def test_xlm(self):
        p = nlu.load("xx.embed.xlm", verbose=True)
        df = p.predict("I love new embeds baby", output_level="token")
        for c in df.columns:
            print(df[c])


if __name__ == "__main__":
    unittest.main()
