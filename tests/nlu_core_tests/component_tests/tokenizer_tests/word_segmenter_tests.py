


import unittest
from nlu import *
class TestWordSegmenter(unittest.TestCase):

    def test_word_segmenter(self):
        pipe = nlu.load('zh.segment_words',verbose=True)
        data = '您的生活就是矩阵编程固有的不平衡方程的剩余部分之和。您是异常的最终结果，尽管做出了我最大的努力，但我仍然无法消除数学精度的和谐。尽管仍然不遗余力地避免了负担，但这并不意外，因此也不超出控制范围。这无可避免地将您引向了这里。'
        df = pipe.predict(data,output_level='token')
        print(df.columns)
        print(df)
        print( df['token'])

        pipe = nlu.load('zh.tokenize',verbose=True)
        data = '您的生活就是矩阵编程固有的不平衡方程的剩余部分之和。您是异常的最终结果，尽管做出了我最大的努力，但我仍然无法消除数学精度的和谐。尽管仍然不遗余力地避免了负担，但这并不意外，因此也不超出控制范围。这无可避免地将您引向了这里。'
        df = pipe.predict([data], output_level='sentence')
        print(df.columns)
        print( df['token'])





if __name__ == '__main__':
    unittest.main()

