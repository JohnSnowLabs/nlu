import unittest
import nlu


class SpanBertCorefCase(unittest.TestCase):
    def test_coref_model(self):
        data = 'John told Mary he would like to borrow a book'
        p = nlu.load('en.coreference.spanbert')
        res = p.predict(data)

        for c in res.columns:
            print(res[c])


if __name__ == '__main__':
    unittest.main()
