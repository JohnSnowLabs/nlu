import unittest
import nlu


class CamemBertForQuestionAnsweringTestCase (unittest.TestCase):
    def test_albert_for_question_answering(self):
        pipe = nlu.load("fr.answer_question.camembert.fquad", verbose=True)
        data = "What is my name?|||My name is CKL"
        df = pipe.predict(
            data,
        )
        for c in df.columns:
            print(df[c])



if __name__ == "__main__":
    unittest.main()
