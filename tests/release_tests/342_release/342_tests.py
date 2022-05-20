import sys
import unittest

import nlu


class Test341(unittest.TestCase):
    def test_341_models(self):
        te = [
            "en.embed.deberta_v3_xsmall",
            "en.embed.deberta_v3_small",
            "en.embed.deberta_v3_base",
            "en.embed.deberta_v3_large",
            "xx.embed.mdeberta_v3_bas",
        ]

        fails = []
        for t in te:
            try:
                print(f"Testing spell = {t}")
                pipe = nlu.load(t, verbose=True)
                df = pipe.predict(
                    ["Peter love pancaces. I hate Mondays", "I love Fridays"]
                )
                for c in df.columns:
                    print(df[c])
            except Exception as err:
                print(f"Failure for spell = {t} ", err)
                e = sys.exc_info()
                print(e[0])
                print(e[1])
                fails.append(t)

        fail_string = "\n".join(fails)
        print(f"Done testing, failures = {fail_string}")
        if len(fails) > 0:
            raise Exception("Not all new spells completed successfully")

    def test_341_HC_models(self):
        import tests.secrets as sct

        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )
        te = [
            "en.med_ner.clinical_trials",
            "es.med_ner.deid.generic.roberta",
            "es.med_ner.deid.subentity.roberta",
            "en.med_ner.deid.generic_augmented",
            "en.med_ner.deid.subentity_augmented",
        ]
        sample_texts = [
            """
            Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020 y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos..
                        """,
            """
                        Datos del paciente. Nombre:  Jose . Apellidos: Aranda Martinez. NHC: 2748903. NASS: 26 37482910.
                        """,
            """The patient was given metformin 500 mg, 2.5 mg of coumadin and then ibuprofen""",
            """he patient was given metformin 400 mg, coumadin 5 mg, coumadin, amlodipine 10 MG""",
            """To compare the results of recording enamel opacities using the TF and modified DDE indices.""",
            """I felt a bit drowsy and had blurred vision after taking Aspirin.""",
        ]
        fails = []
        succs = []
        for t in te:
            try:
                print(f"Testing spell = {t}")
                pipe = nlu.load(t, verbose=True)
                df = pipe.predict(
                    sample_texts,
                    drop_irrelevant_cols=False,
                    metadata=True,
                )
                print(df.columns)
                for c in df.columns:
                    print(df[c])
                succs.append(t)
            except Exception as err:
                print(f"Failure for spell = {t} ", err)
                fails.append(t)
                break
        fail_string = "\n".join(fails)
        succ_string = "\n".join(succs)
        print(f"Done testing, failures = {fail_string}")
        print(f"Done testing, successes = {succ_string}")
        if len(fails) > 0:
            raise Exception("Not all new spells completed successfully")

    def test_quick(self):
        import tests.secrets as sct

        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )
        te = [
            "sentiment",
        ]
        sample_texts = ["""Billy loves Soda. Sarah said so"""]
        succs = []
        for t in te:
            print(f"Testing spell = {t}")
            pipe = nlu.load(t, verbose=True)
            df = pipe.predict(
                sample_texts,
                drop_irrelevant_cols=False,
                metadata=True,
                output_level="document",
            )

            print(df.columns)
            for c in df.columns:
                print(df[c])
            succs.append(t)


if __name__ == "__main__":
    unittest.main()
