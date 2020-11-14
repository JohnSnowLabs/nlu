


import unittest
from nlu import *
class TestCyber(unittest.TestCase):

    def test_cyber_model(self):
        pipe = nlu.load('cyberbullying',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence')
        print(df.columns)
        print(df['sentence'], df[['cyberbullying','cyberbullying_confidence']])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        self.assertIsInstance(df.iloc[0]['cyberbullying'],str )
        print(df.columns)
        print(df['document'], df[['cyberbullying','cyberbullying_confidence']])
        self.assertIsInstance(df.iloc[0]['cyberbullying'], str)

if __name__ == '__main__':
    unittest.main()

