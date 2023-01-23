import unittest

import nlu

class TestBertSeqClassifier(unittest.TestCase):

    def test_camembert_seq(self):
        pipe = nlu.load("fr.camembert_base_sequence_classifier_allocine_TEST", verbose=True)
        data = "Bill Gates and Steve Jobs are good friends"
        df = pipe.predict([data], output_level="document")
        for c in df.columns:
            print(df[c])

if __name__ == "__main__":
    unittest.main()
