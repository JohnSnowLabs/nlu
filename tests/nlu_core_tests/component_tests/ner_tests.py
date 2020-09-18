import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestNer(unittest.TestCase):

    def test_ner_pipe(self):
        print("MY TEST")
        df = nlu.load('ner', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='chunk' )
        print(df.columns)
        print(df[[]])
        print(df[[ 'entities', 'ner_tag']])


        df = nlu.load('ner', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='document' )
        print(df.columns)
        print(df[[ 'entities', 'ner_tag']])


        df = nlu.load('ner', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='sentence' )
        print(df.columns)
        print(df[[ 'entities', 'ner_tag']])


        df = nlu.load('ner', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='token' )
        print(df.columns)
        print(df[['entities', 'ner_tag']])

        #xx
        df = nlu.load('en.ner.onto.glove.6B_100d', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='chunk' )
        print(df.columns)
        print(df[[ 'entities', 'ner_tag']])


        df = nlu.load('en.ner.onto.glove.6B_100d', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='document' )
        print(df.columns)
        print(df[[ 'entities', 'ner_tag']])


        df = nlu.load('en.ner.onto.glove.6B_100d', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='sentence' )
        print(df.columns)
        print(df[[ 'entities', 'ner_tag']])


        df = nlu.load('en.ner.onto.glove.6B_100d', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='token' )
        print(df.columns)
        print(df[[ 'entities', 'ner_tag']])




if __name__ == '__main__':
    unittest.main()

