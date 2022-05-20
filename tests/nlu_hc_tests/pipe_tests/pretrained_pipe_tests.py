import unittest

import nlu
import tests.secrets as sct


class TestPretrainedPipe(unittest.TestCase):
    def test_pretrained_pipe(self):

        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )
        data = [
            "The patient has cancer and high fever and will die next week.",
            " She had a seizure.",
        ]
        res = nlu.load("en.explain_doc.era", verbose=True).predict(data)

        for c in res:
            print(res[c])

        print(res)


if __name__ == "__main__":
    TestPretrainedPipe().test_pretrained_pipe()
