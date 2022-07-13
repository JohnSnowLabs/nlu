import unittest
import nlu


class DeBertaForQuestionAnsweringTestCase (unittest.TestCase):
    def test_deberta_for_question_answering(self):
        pipe = nlu.load("en.answer_question.squadv2.deberta", verbose=True)
        data = "What is my name?|||My name is CKL"
        df = pipe.predict(
            data,
        )
        for c in df.columns:
            print(df[c])



if __name__ == "__main__":
    unittest.main()
