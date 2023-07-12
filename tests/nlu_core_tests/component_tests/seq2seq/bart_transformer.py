import os
import sys
import unittest
from nlu import *


summarizer_spells = [
    'en.seq2seq.distilbart_cnn_6_6',
    'en.seq2seq.distilbart_xsum_12_6'
]

class BartTransformerTests(unittest.TestCase):
    def test_bart_transformer(self):


        for s in summarizer_spells:
            pipe = nlu.load(s)
            # Configure relations to extract
            print("TESTING: ", s)
            df = pipe.predict("Paracetamol can alleviate headache or")
            print(df.columns)
            for c in df:
                print(c)
                print(df[c])

            print(df)

if __name__ == "__main__":
    unittest.main()
