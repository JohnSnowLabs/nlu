


import unittest
from nlu import *
class TestMarian(unittest.TestCase):
    def test_marian_en_to_de(self):
        pipe = nlu.load('en.translate_to.de',verbose=True)
        data = ['Who is president of germany', 'Who is donald trump ?', 'What is NLP?', 'How to make tea?']
        df = pipe.predict(data, output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
        print(df.columns)

        print(df['translation'])
        print(df.columns)

    def test_marian_de_to_en(self):
        pipe = nlu.load('de.translate_to.en',verbose=True)
        # test for each tasks
        data = ['Wer ist Praesident von Deutschland', 'Wer ist donald trump ?', 'Was ist NLP?', 'Wie macht man Tee?']
        df = pipe.predict(data)
        print(df.columns)
        print(df['translation'])
        print(df.columns)

    def test_marian_de_to_en_pipe(self):
        pipe = nlu.load('de.marian.translate_to.en',verbose=True)
        print('QQP')
        # pipe.print_info()
        # pipe['t5'].setTask('Answer the question')

        # test for each tasks
        data = ['Wer ist Praesident von Deutschland', 'Wer ist donald trump ?', 'Was ist NLP?', 'Wie macht man Tee?']
        df = pipe.predict(data)
        print(df.columns)
        print(df['marian'])
        print(df.columns)



if __name__ == '__main__':
    unittest.main()

