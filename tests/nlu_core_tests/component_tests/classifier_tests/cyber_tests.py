

import unittest
from nlu import *
class TestCyber(unittest.TestCase):


    def test_cyber_model(self):
        import pandas as pd
        pipe = nlu.load('cyberbullying',verbose=True)
        df = pipe.predict(['Peter love pancaces. I hate Mondays', 'I love Fridays'], output_level='token',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])

if __name__ == '__main__':
    unittest.main()

