


import unittest
from nlu import *

class TestQuestions(unittest.TestCase):

    def test_questions_model(self):
        pipe = nlu.load('questions',verbose=True)
        data = ['I love pancaces. I hate Mondays', 'I love Fridays']
        df = pipe.predict(data, output_level='sentence')
        for c in df.columns: print(df[c])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        for c in df.columns: print(df[c])





if __name__ == '__main__':
    unittest.main()

