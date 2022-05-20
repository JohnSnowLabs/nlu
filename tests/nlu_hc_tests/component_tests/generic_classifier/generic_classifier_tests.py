import unittest

import nlu
import tests.secrets as sct


class DeidentificationTests(unittest.TestCase):
    def test_generic_classifier(self):

        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET

        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )
        #

        res = nlu.load("bert elmo", verbose=True).predict(
            "DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin"
        )

        # elmo_embeddings and bert_embeddings   is what should be passed 2 the feature asselmber/generic classifier

        # res.show()
        # for os_components in res.columns:
        #     print(os_components)
        #     res.select(os_components).show(truncate=False)
        # res = nlu.load('en.extract_relation', verbose=True).predict('The patient got cancer in my foot and damage in his brain')

        for c in res:
            print(c)
            print(res[c])

        # print(res)


if __name__ == "__main__":
    DeidentificationTests().test_entities_config()
