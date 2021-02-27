


import unittest
from nlu import *
class TestCyber(unittest.TestCase):

    def test_snips_classifer_model(self):
        pipe = nlu.load('en.classify.snips',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'])
        print(df.columns)
        for c in df.columns:print(c,df[c])

    def test_snips_ner_model(self):
        pipe = nlu.load('en.ner.snips',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'])
        print(df.columns)
        for c in df.columns:print(c,df[c])

    def test_quick(self):
        # pipe = nlu.load('bn.ner.cc_300d',verbose=True)
        # pipe = nlu.load('bn.embed',verbose=True)
        pipe = nlu.load('en.ner.snips',verbose=True)

        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'])
        print(df.columns)
        for c in df.columns:print(c,df[c])

if __name__ == '__main__':
    unittest.main()

