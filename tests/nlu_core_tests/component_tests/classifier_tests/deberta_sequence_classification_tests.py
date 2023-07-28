import unittest

import nlu


class TestDebertaSeqClassifier(unittest.TestCase):
    def test_deberta_seq_classifier(self):

        models = [
            #"en.classify.sentiment.imdb.deberta.base",
            #"en.classify.sentiment.imdb.deberta.large",
            "en.classify.news.deberta.small",
            #"en.classify.dbpedia",
            #"en.classify.sentiment.imdb.deberta.small",
            #"en.classify.news.deberta",
            #"en.classify.sentiment.imdb.deberta",
            #"fr.classify.allocine",
            #"ur.classify.sentiment.imdb"
        ]

        for model in models:
            pipe = nlu.load(model, verbose=True)
            df = pipe.predict(
                ["I really liked that movie!"],
                output_level="document",
                drop_irrelevant_cols=False,
            )
            for c in df.columns:
                print(df[c])


if __name__ == "__main__":
    unittest.main()
