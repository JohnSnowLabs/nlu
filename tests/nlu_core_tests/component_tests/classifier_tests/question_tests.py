


import unittest
from nlu import *

class TestQuestions(unittest.TestCase):

    def test_questions_model(self):
        pipe = nlu.load('questions',verbose=True)
        data = ['I love pancaces. I hate Mondays', 'I love Fridays']
        df = pipe.predict(data, output_level='sentence')
        print(df.columns)
        print(df['sentence'], df[['questions','questions_confidence']])
        df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='document')
        self.assertIsInstance(df.iloc[0]['questions'],str )
        print(df.columns)
        print(df['document'], df[['questions','questions_confidence']])
        self.assertIsInstance(df.iloc[0]['questions'], str)

        # pipe = nlu.load('questions',verbose=True)
        # df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='token')
        # print(df.columns)
        # print(df['sentence'], df[['questions','questions_confidence']])
        # df = pipe.predict(['I love pancaces. I hate Mondays', 'I love Fridays'], output_level='chunk')
        # self.assertIsInstance(df.iloc[0]['questions'],str )
        # print(df.columns)
        # print(df['document'], df[['questions','questions_confidence']])
        # self.assertIsInstance(df.iloc[0]['questions'], str)


    def test_quick(self):
        r = nlu.load('en.classify.questions').predict('How expensive is the Watch?')
        print(r)


if __name__ == '__main__':
    unittest.main()

