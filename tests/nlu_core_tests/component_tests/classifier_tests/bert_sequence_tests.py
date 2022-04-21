import unittest

import nlu


class TestBertSeqClassifier(unittest.TestCase):
    def test_bert_seq_classifier(self):

        te = [
            #
            # 'en.classify.bert_sequence.imdb_large',
            #  'en.classify.bert_sequence.imdb',
            #     'en.classify.bert_sequence.ag_news',
            #     'en.classify.bert_sequence.dbpedia_14',
            #     'en.classify.bert_sequence.finbert',
            "en.classify.bert_sequence.dehatebert_mono",
        ]

        for t in te:
            pipe = nlu.load(t, verbose=True)
            df = pipe.predict(
                ["Peter love pancaces. I hate Mondays", "I love Fridays"],
                output_level="document",
                drop_irrelevant_cols=False,
            )
            for c in df.columns:
                print(df[c])


if __name__ == "__main__":
    unittest.main()
