import unittest
import nlu


class XlmRoBertaForQuestionAnsweringTestCase (unittest.TestCase):
    def test_xlmroberta_for_question_answering(self):
        pipe = nlu.load("en.answer_question.squadv2.xlm_roberta.base", verbose=True)
        data = "What is my name?|||My name is CKL"
        df = pipe.predict(
            data,
        )
        for c in df.columns:
            print(df[c])



if __name__ == "__main__":
    unittest.main()
