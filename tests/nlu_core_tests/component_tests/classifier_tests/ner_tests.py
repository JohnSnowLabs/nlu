import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *

class TestNer(unittest.TestCase):

    # def test_ner_tok_bug(self):
    #     text =  ['Zebra stripes and their role in flies']
    #     d = nlu.load('ner', verbose=True).predict(text)
    #     print(d)
    #     print(d.columns)


    # def test_ner_pipe(self):
    #     print("CHUNK")
    #     df = nlu.load('en.ner.onto.glove.6B_100d', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='chunk' ,metadata=True)
    #     for c in df.columns: print(df[c])

        #
        # print("DOCUMENT")
        # df = nlu.load('en.ner.onto.glove.6B_100d', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='document',metadata=True )
        # print(df.columns)
        # print(df[[ 'entities', 'document']])
        # print(df[[ 'entities', 'entities_confidence']])
        # print(df[[ 'entities', 'ner_confidence']])
        # print(df[[ 'document', 'ner_confidence']])
        # print("SENTENCE")
        # df = nlu.load('en.ner.onto.glove.6B_100d', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='sentence' ,metadata=True)
        # print(df.columns)
        # print(df[[ 'entities', 'sentence']])
        # print(df[[ 'entities', 'entities_confidence']])
        # print(df[[ 'entities', 'ner_confidence']])
        # print(df[[ 'sentence', 'ner_confidence']])
        #
        #
        # print("TOKEN")
        # df = nlu.load('en.ner.onto.glove.6B_100d', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='token' ,metadata=True)
        # print(df.columns)
        # print(df[[ 'entities', 'entities_confidence']])
        # print(df[[ 'entities', 'ner_confidence']])
        # print(df[[ 'token', 'ner_confidence']])
        # print(df[[ 'ner', 'ner_confidence']])



    def test_zh_ner(self):
        pipe = nlu.load('zh.ner')
        data = '您的生活就是矩阵编程固有的不平衡方程的剩余部分之和。您是异常的最终结果，尽管做出了我最大的努力，但我仍然无法消除数学精度的和谐。尽管仍然不遗余力地避免了负担，但这并不意外，因此也不超出控制范围。这无可避免地将您引向了这里。'
        df = pipe.predict([data], output_level='document')
        for c in df.columns: print(df[c])


    def test_aspect_ner(self):
        pipe = nlu.load('en.ner.aspect_sentiment')
        data = 'We loved our Thai-style main which amazing with lots of flavours very impressive for vegetarian. But the service was below average and the chips were too terrible to finish.'
        df = pipe.predict([data], output_level='document')
        for c in df.columns: print(df[c])


    def test_ner_pipe_confidences(self):
        #
        df = nlu.load('en.ner.onto.glove.6B_100d', verbose=True ).predict('Donald Trump from America and Angela Merkal from Germany dont share many oppinions.', output_level='token', metadata=True)
        for c in df.columns: print(df[c])

if __name__ == '__main__':
    unittest.main()

