

import unittest
from nlu import *
class TestCyber(unittest.TestCase):

    def test_quick(self):
        p = '/tmp/i2b2_clinical_rel_dataset.csv'
        import pandas as pd
        df =pd.read_csv(p)

        # res = nlu.load('toxic',verbose=True).predict(" I LOCVE PEANUT BUTTEr. AND YELLY. AND FUCK YOU BITCH OK !@:!?!??!?!", output_level='document')
        res =  nlu.load('en.classify.questions').predict('How expensive is the Watch? Whats the fastest way to Berlin?',output_level='sentence')
        print(res.columns)
        for c in res.columns : print(res[c])



def test_cyber_model(self):
        import pandas as pd
        pipe = nlu.load('sentiment',verbose=True)
        df = pipe.predict(['Peter love pancaces. I hate Mondays', 'I love Fridays'], output_level='token',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])

if __name__ == '__main__':
    unittest.main()

