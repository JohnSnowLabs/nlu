import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class MatchTests(unittest.TestCase):

    def test_pattern_matcher(self):
        pass
        pipe = nlu.load('match.pattern', verbose=True )
        df = pipe.predict('2020 was a crazy year but wait for October 1. 2020')
        for c in df.columns: print(df[c])


    def test_chunk_matcher(self):
        pass
        pipe = nlu.load('match.chunks', verbose=True )
        df = pipe.predict('2020 was a crazy year but wait for October 1. 2020')
        for c in df.columns: print(df[c])

    def download_entities_files(self):
        import urllib2
        response = urllib2.urlopen('https://wordpress.org/plugins/about/readme.txt')
        data = response.read()
        filename = "readme.txt"
        file_ = open(filename, 'w')
        file_.write(data)
        file_.close()


    def test_text_matcher(self):
        p = '/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tmp/trasgh/entities.txt'
        pipe = nlu.load('match.text', verbose=True )
        pipe['text_matcher'].setEntities(p)
        df = pipe.predict('2020 was a crazy year but wait for October 1. 2020')
        for c in df.columns: print(df[c])
    def test_regex_matcher(self):
        p='/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tmp/trasgh/rulesd.txt'
        pipe = nlu.load('match.regex', verbose=True )
        pipe['regex_matcher'].setExternalRules(path=p, delimiter=',')
        df = pipe.predict('2020 was a crazy year but wait for October 1. 2020')
        for c in df.columns: print(df[c])
    def test_date_matcher(self):
        pipe = nlu.load('match.date', verbose=True )
        df = pipe.predict('2020 was a crazy year but wait for October 1. 2020')
        for c in df.columns: print(df[c])

if __name__ == '__main__':
    unittest.main()

