import unittest
import nlu
import sys
import tests.secrets as sct

class Test401(unittest.TestCase):



    def test_401_models(self):
        import pandas as pd
        q = 'What is my name?'
        c = 'My name is Clara and I live in Berkeley'


        te = ['de.embed.electra.base_64d_700000_cased.by_stefan_it',
        'en.ner.pubmed_bert.chemical_pubmed_biored.512d_5_modified.by_ghadeermobasher',
        'fr.embed.camem_bert.by_katrin_kc',
        'xx.answer_question.multi_lingual_bert.mlqa.finetuned.extra_support_hi_zh',
        'zh.ner.pos.base.by_ckiplab']
        data = f'{q}|||{c}'
        data = [data,data,data]
        fails = []
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
        fail_string = "\n".join(fails)
        print(f'Done testing, failures = {fail_string}')
        if len(fails) > 0:
            raise Exception("Not all new spells completed successfully")


    def test_401_HC_models(self):
        import tests.secrets as sct
        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET)
        te = [
            'en.classify.adverse_drug_events',
             'en.classify.health_stance',
             'en.clasify.health_premise',
             'en.classify.treatment_sentiment',
             'en.classify.drug_reviews',
             'en.classify.self_reported_age',
             'es.classify.self_reported_symptoms',
             'en.classify.self_reported_vaccine_status',
             'en.classify.self_reported_partner_violence',
             'en.classify.exact_age',
             'en.classify.self_reported_stress',
             'en.classify.health_mentions',
             'en.classify.health',
             'en.classify.bert_sequence_vaccine_sentiment',
             'en.classify.vaccine_sentiment',
             'en.classify.stressor',
             'en.ner.medication',
             'en.resolve.medication',
             'es.classify.disease_mentions',
             'en.map_entity.umls_clinical_drugs_mapper',
             'en.map_entity.umls_clinical_findings_mapper',
             'en.map_entity.umls_disease_syndrome_mapper',
             'en.map_entity.umls_major_concepts_mapper',
             'en.map_entity.umls_drug_substance_mapper',
             'en.map_entity.umls_drug_resolver',
             'en.map_entity.umls_clinical_findings_resolver',
             'en.map_entity.umls_disease_syndrome_resolver',
             'en.map_entity.umls_major_concepts_resolver',
             'en.map_entity.umls_drug_substance_resolver',
             'en.relation.adverse_drug_events.conversational',
             'en.de_identify.clinical_slim',
             'en.de_identify.clinical_pipeline',
             'en.classify.token_bert.anatem',
             'en.classify.token_bert.bc2gm_gene',
             'en.classify.token_bert.bc4chemd_chemicals',
             'en.classify.token_bert.bc5cdr_chemicals',
             'en.classify.token_bert.bc5cdr_disease',
             'en.classify.token_bert.jnlpba_cellular',
             'en.classify.token_bert.linnaeus_species',
             'en.classify.token_bert.ncbi_disease',
             'en.classify.token_bert.species',
             'en.classify.token_bert.pathogen']

        
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
        fail_string = '\n'.join(fails)
        succ_string = '\n'.join(succs)
        print(f'Done testing, failures = {fail_string}')
        print(f'Done testing, successes = {succ_string}')
        if len(fails) > 0:
            raise Exception("Not all new spells completed successfully")


if __name__ == '__main__':
    unittest.main()
