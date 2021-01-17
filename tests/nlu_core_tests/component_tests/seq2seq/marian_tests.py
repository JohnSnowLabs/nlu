


import unittest
from nlu import *
class TestMarian(unittest.TestCase):
# GENERATE NAME SPACE ENTRIES PROGRAMMATICLY?!??
    def test_marian_en_to_de(self):
        pipe = nlu.load('en.translate_to.de',verbose=True)
        print('QQP')
        # pipe.print_info()
        # pipe['t5'].setTask('Answer the question')

        # TODO DEFAULT T5 INFERENCE TO DOC level
        # test for each tasks
        data = ['Who is president of germany', 'Who is donald trump ?', 'What is NLP?', 'How to make tea?']
        df = pipe.predict(data)
        print(df.columns)

        print(df['translation'])
        print(df.columns)

    def test_marian_de_to_en(self):
        pipe = nlu.load('de.translate_to.en',verbose=True)
        print('QQP')
        # pipe.print_info()
        # pipe['t5'].setTask('Answer the question')

        # TODO DEFAULT T5 INFERENCE TO DOC level
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

        # TODO DEFAULT T5 INFERENCE TO DOC level
        # test for each tasks
        data = ['Wer ist Praesident von Deutschland', 'Wer ist donald trump ?', 'Was ist NLP?', 'Wie macht man Tee?']
        df = pipe.predict(data)
        print(df.columns)
        print(df['marian'])
        print(df.columns)


    def test_quick(self):
        matrix_quotes = [
            'You are here because Zion is about to be destroyed. Its every living inhabitant terminated, its entire existence eradicated.',
            'Denial is the most predictable of all human responses. But, rest assured, this will be the sixth time we have destroyed it, and we have become exceedingly efficient at it.',
            'Your life is the sum of a remainder of an unbalanced equation inherent to the programming of the matrix. You are the eventuality of an anomaly, which despite my sincerest efforts I have been unable to eliminate from what is otherwise a harmony of mathematical precision. While it remains a burden assiduously avoided, it is not unexpected, and thus not beyond a measure of control. Which has led you, inexorably, here.',
            'Hope, it is the quintessential human delusion, simultaneously the source of your greatest strength, and your greatest weakness.',
            'Dont Think You Are, Know You Are.',
            'As you were undoubtedly gathering, the anomaly is systemic, creating fluctuations in even the most simplistic equations.',
            'If I am the father of the Matrix, she would undoubtedly be its mother.',]

        translate_pipe = nlu.load('en.translate_to.de')
        de_df = translate_pipe.predict(matrix_quotes, output_level='document')
        print( de_df[['document','translation']])


if __name__ == '__main__':
    unittest.main()

