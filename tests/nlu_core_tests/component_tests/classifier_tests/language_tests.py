


import unittest
from nlu import *

class TestLanguage(unittest.TestCase):

    def test_language_model(self):
        pipe = nlu.load('lang',verbose=True)
        data = ['NLU is an open-source text processing library for advanced natural language processing for the Python language.',
                'NLU est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python.',
                'NLU ist eine Open-Source Text verarbeitungs Software fuer fortgeschrittene natuerlich sprachliche Textverarbeitung in der Python Sprache '
                ]
        df = pipe.predict(data, output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])
        df = pipe.predict(data, output_level='document')
        for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()

