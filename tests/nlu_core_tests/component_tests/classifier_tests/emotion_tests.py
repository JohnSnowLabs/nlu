


import unittest
from nlu import *

class TestEmotion(unittest.TestCase):

    def test_emotion_model(self):
        #TODO bad output level inference for some model!2
        ## todo bug, if we Input a row with 2 sentences, where first sentence is happy and second is sad,
        # NLU will predict both as happy. If you reverse order both become sad

        pipe = nlu.load('emotion',verbose=True)
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence')
        print(df.columns)
        print(df['sentence'], df[['emotion','emotion_confidence']])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        self.assertIsInstance(df.iloc[0]['emotion'],str )
        print(df.columns)
        print(df['document'], df[['emotion','emotion_confidence']])
        self.assertIsInstance(df.iloc[0]['emotion'], str)


if __name__ == '__main__':
    unittest.main()

