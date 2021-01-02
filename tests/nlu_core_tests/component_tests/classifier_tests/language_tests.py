


import unittest
from nlu import *

class TestLanguage(unittest.TestCase):

    def test_language_model(self):
        pipe = nlu.load('lang',verbose=True)
        data = ['NLU is an open-source text processing library for advanced natural language processing for the Python language.',
                'NLU est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python.',
                'NLU ist eine Open-Source Text verarbeitungs Software fuer fortgeschrittene natuerlich sprachliche Textverarbeitung in der Python Sprache '
                ]
        df = pipe.predict(data, output_level='sentence')
        print(df.columns)
        print(df['sentence'], df[['language','language_confidence']])
        self.assertIsInstance(df.iloc[0]['language'],str )
        df = pipe.predict(data, output_level='document')
        print(df.columns)
        print(df['document'], df[['language','language_confidence']])
        self.assertIsInstance(df.iloc[0]['language'], str)

if __name__ == '__main__':
    unittest.main()

