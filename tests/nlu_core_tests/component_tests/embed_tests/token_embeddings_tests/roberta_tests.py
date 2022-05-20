import unittest

from nlu import *


class TestRobertaEmbeddings(unittest.TestCase):
    def test_roberta(self):
        embeds = [
            "en.embed.distilbert",
            "en.embed.distilbert.base",
            "en.embed.distilbert.base.uncased",
            "en.embed.distilroberta",
            "en.embed.roberta",
            "en.embed.roberta.base",
            "en.embed.roberta.large",
            "xx.embed.distilbert.",
            "xx.embed.xlm",
            "xx.embed.xlm.base",
            "xx.embed.xlm.twitter",
        ]
        for e in embeds:
            print(
                f"+++++++++++++++++++++++++++++++++++++++++++++++++{e}+++++++++++++++++++++++++++++++++++++++++++++++++"
            )
            p = nlu.load("en.embed.roberta")
            df = p.predict("I love new embeds baby", output_level="token")
            for c in df.columns:
                print(df[c])

    def test_new_embeds(self):
        embeds = [
            "en.embed.distilbert",
            "en.embed.distilbert.base",
            "en.embed.distilbert.base.uncased",
            "en.embed.distilroberta",
            "en.embed.roberta",
            "en.embed.roberta.base",
            "en.embed.roberta.large",
            "xx.embed.distilbert.",
            "xx.embed.xlm",
            "xx.embed.xlm.base",
            "xx.embed.xlm.twitter",
        ]
        for e in embeds:
            print(
                f"+++++++++++++++++++++++++++++++++++++++++++++++++{e}+++++++++++++++++++++++++++++++++++++++++++++++++"
            )
            p = nlu.load(e, verbose=True)
            df = p.predict("I love new embeds baby", output_level="token")
            for c in df.columns:
                print(df[c])


if __name__ == "__main__":
    unittest.main()
