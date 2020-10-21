


import unittest
from nlu import *

class TestSpam(unittest.TestCase):

    def test_spam_model(self):
        pipe = nlu.load('spam',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence')
        print(df.columns)
        print(df['sentence'], df[['spam','spam_confidence']])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        self.assertIsInstance(df.iloc[0]['spam'],str )
        print(df.columns)
        print(df['document'], df[['spam','spam_confidence']])
        self.assertIsInstance(df.iloc[0]['spam'], str)



if __name__ == '__main__':
    unittest.main()

