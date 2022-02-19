import unittest
import nlu


class TestSpellCheckers(unittest.TestCase):

    def test_spell_context(self):
        pipe = nlu.load('en.spell', verbose=True)
        df = pipe.predict('I liek penut buttr and jelly', drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])

    def test_spell_sym(self):
        component_list = nlu.load('spell.symmetric', verbose=True)
        df = component_list.predict('I liek penut buttr and jelly', drop_irrelevant_cols=False, metadata=True, )
        for os_components in df.columns: print(df[os_components])

    def test_spell_norvig(self):
        component_list = nlu.load('spell.norvig', verbose=True)
        df = component_list.predict('I liek penut buttr and jelly', drop_irrelevant_cols=False, metadata=True, )
        for os_components in df.columns: print(df[os_components])


if __name__ == '__main__':
    unittest.main()
