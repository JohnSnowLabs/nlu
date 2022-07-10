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

    def test_400_HC_models(self):
        import tests.secrets as sct
        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET)
        te = [
            'en.med_ner.clinical_trials_abstracts',
'en.med_ner.bert_clinical_trials_abstracts',
'en.med_ner.pathogen',
'en.med_ner.bert_living_species',
'en.med_ner.living_species',
'en.med_ner.biobert_living_species',
'en.classify.bert_stress',
'es.med_ner.bert_living_species',
'es.med_ner.living_species_bert',
'es.med_ner.living_species_roberta',
'es.med_ner.living_species_300',
'es.med_ner.living_species',
'fr.med_ner.living_species',
'fr.med_ner.bert_living_species',
'pt.med_ner.token_bert_living_species',
'pt.med_ner.living_species',
'pt.med_ner.living_species_roberta',
'pt.med_ner.bert_living_species',
'it.med_ner.bert_living_species',
'it.med_ner.living_species_bert',
'it.med_ner.living_species',
'ro.med_ner.bert_living_species',
'ro.med_ner.clinical',
'ro.med_ner.clinical_bert',
'ro.med_ner.deid.subentity',
'ro.med_ner.deid.subentity.bert',
'ca.med_ner.living_species',
'gl.med_ner.living_species'
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


if __name__ == '__main__':
    unittest.main()
