import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestSpellCheckers (unittest.TestCase):

    def test_spell_context(self):
        pipe = nlu.load('spell', verbose=True )
        df = pipe.predict('I liek penut butter and jellli', output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])
    #
    # def test_spell_sym(self):
    #     pipe = nlu.load('spell.symmetric', verbose=True )
    #     df = pipe.predict('I liek penut butter and jellli', output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
    #     for c in df.columns: print(df[c])
    #
    # def test_spell_norvig(self):
    #     pipe = nlu.load('spell.norvig', verbose=True )
    #     df = pipe.predict('I liek penut butter and jellli', output_level='sentence',drop_irrelevant_cols=False, metadata=True, )
    #     for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()

