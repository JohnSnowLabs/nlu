import json
import os
import unittest

import nlu
import tests.secrets as sct


class TestAuthentification(unittest.TestCase):
    def test_auth(self):
        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        # JSL_SECRET            = sct.JSL_SECRET_3_4_2
        JSL_SECRET = sct.JSL_SECRET
        res = (
            nlu.auth(
                SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
            )
            .load("en.med_ner.diseases")
            .predict("He has cancer")
        )
        for c in res.columns:
            print(res[c])

    def test_auth_miss_match(self):
        SPARK_NLP_LICENSE = "wrong_license"
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        # JSL_SECRET            = sct.JSL_SECRET_3_4_2
        JSL_SECRET = sct.JSL_SECRET
        res = (
            nlu.auth(
                SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
            )
            .load("en.med_ner.diseases")
            .predict("He has cancer")
        )
        for c in res.columns:
            print(res[c])

    def test_auth_via_file(self):
        secrets_json_path = os.path.join(os.path.abspath("./"), "license.json")
        print("license path:", secrets_json_path)
        with open(secrets_json_path, "w", encoding="utf8") as file:
            json.dump(sct.license_dict, file)
        res = (
            nlu.auth(secrets_json_path)
            .load("en.med_ner.diseases", verbose=True)
            .predict("He has cancer")
        )
        # res = nlu.load('en.med_ner.diseases',verbose=True).predict("He has cancer")
        for c in res.columns:
            print(res[c])


if __name__ == "__main__":
    unittest.main()
