import os
import sys
import unittest

from nlu import *


summarizer_spells = [
    'en.summarize_distilbart.cnn_.6.6'
]

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

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
