


import unittest
from nlu import *

class TestLanguage(unittest.TestCase):

    def test_language_model(self):
        pipe = nlu.load('lang',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence')
        print(df.columns)
        print(df['sentence'], df[['language','language_confidence']])
        self.assertIsInstance(df.iloc[0]['language'],str )
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        print(df.columns)
        print(df['document'], df[['language','language_confidence']])
        self.assertIsInstance(df.iloc[0]['language'], str)

if __name__ == '__main__':
    unittest.main()

