


import unittest
from nlu import *

class TestEmotion(unittest.TestCase):

    def test_emotion_model(self):
        # NLU will predict both as happy. If you reverse order both become sad

        pipe = nlu.load('emotion',verbose=True)
        # df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
        # for c in df.columns: print(df[c])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()

