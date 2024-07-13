import os
import sys
import unittest
from nlu import *


class M2M100TransformerTests(unittest.TestCase):
    def test_m2m100_transformer(self):
            model = nlu.load("xx.m2m100_418M")
            df = model.predict("生活就像一盒巧克力。")
            print(df.columns)

if __name__ == "__main__":
    unittest.main()
