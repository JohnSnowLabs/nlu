import unittest
import nlu



def test_albert_for_question_answering():
    pipe = nlu.load("en.answer_question.squadv2.albert.xxl.by_sultan", verbose=True)
    data = "What is my name?|||My name is CKL"
    df = pipe.predict(
        data,
    )
    for c in df.columns:
        print(df[c])


