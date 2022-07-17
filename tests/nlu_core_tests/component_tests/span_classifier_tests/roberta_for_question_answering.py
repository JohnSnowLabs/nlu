import unittest
import nlu


class RoBertaForQuestionAnsweringTestCase (unittest.TestCase):
    def test_roberta_for_question_answering(self):
        pipe = nlu.load("en.answer_question.squadv2.roberta.base.by_deepset", verbose=True)
        data = "What is my name?|||My name is CKL"
        df = pipe.predict(
            data,
        )
        for c in df.columns:
            print(df[c])



if __name__ == "__main__":
    unittest.main()
