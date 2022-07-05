import unittest
import nlu
import sys


class Test400(unittest.TestCase):
    def test_400_models(self):
        import pandas as pd
        q = 'What is my name?'
        c = 'My name is Clara and I live in Berkeley'


        te = [
            'en.span_question.albert'
        ]
        data = f'{q}|||{c}'
        data = [data,data,data]
        fails = []
        fail_insta = True
        for t in te:
            try:
                print(f'Testing spell = {t}')
                pipe = nlu.load(t, verbose=True)
                df = pipe.predict(data,metadata=True)
                for c in df.columns: print(df[c])
            except Exception as err:
                print(f'Failure for spell = {t} ', err)
                e = sys.exc_info()
                print(e[0])
                print(e[1])
                fails.append(t)
                if fail_insta :
                    raise Exception(err)
        fail_string = "\n".join(fails)
        print(f'Done testing, failures = {fail_string}')
        if len(fails) > 0:
            raise Exception("Not all new spells completed successfully")

    def test_344_HC_models(self):
        import tests.secrets as sct

        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET)
        te = [
            'en.med_ner.biomedical_bc2gm', # OK?
            'en.resolve.rxnorm_action_treatment', # OK?

            # 'en.classify.rct_binary.use', # BAD
            # 'en.classify.rct_binary.biobert', # BAD
            'pt.med_ner.deid.subentity',
            'pt.med_ner.deid.generic',
            'pt.med_ner.deid',
        ]
        sample_texts = ["""
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
        fail_insta = True
        for t in te:

            try:
                print(f'Testing spell = {t}')
                pipe = nlu.load(t, verbose=True)
                df = pipe.predict(sample_texts, drop_irrelevant_cols=False, metadata=True, )
                print(df.columns)
                for c in df.columns:
                    print(df[c])
                succs.append(t)
            except Exception as err:
                print(f'Failure for spell = {t} ', err)
                fails.append(t)
                if fail_insta:
                    break
        fail_string = '\n'.join(fails)
        succ_string = '\n'.join(succs)
        print(f'Done testing, failures = {fail_string}')
        print(f'Done testing, successes = {succ_string}')
        if len(fails) > 0:
            raise Exception("Not all new spells completed successfully")


    def test_nlu(self):
        import tests.secrets as sct

        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET)
        t = """The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby-girl also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."""
        p = nlu.load('en.classify.token_bert.ner_jsl')
        df = p.predict(t,return_spark_df=True)
        print(df)

        import pyspark.sql.functions as F
        df.select(F.explode(F.arrays_zip('entities@ner_jsl.metadata','entities@ner_jsl.result'))) \
            .select(
            F.expr('col.metadata.entity').alias('entity_label'),
            F.expr('col.result').alias('entity'),
        ).show(n=3,truncate=False)



if __name__ == '__main__':
    unittest.main()
