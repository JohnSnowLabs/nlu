import unittest
import nlu
import sys


class ChunkMapperTestCase(unittest.TestCase):
    import tests.secrets as sct
    SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
    AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
    JSL_SECRET = sct.JSL_SECRET
    nlu.auth(SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET)

    def test_chunk_mapper_for_entities(self):
        # https://nlp.johnsnowlabs.com/2022/06/28/drug_action_treatment_mapper_en_3_0.html
        # https://nlp.johnsnowlabs.com/2022/06/26/abbreviation_mapper_en_3_0.html
        #
        text = """Gravid with estimated fetal weight of 6-6/12 pounds. LABORATORY DATA: Laboratory tests include a CBC which is normal. HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet"""
        res = nlu.load('en.med_ner.abbreviation_clinical en.map_entity.abbreviation_to_definition').predict(text)
        for c in res.columns:
            print(res[c])


    def test_chunk_mapper_for_action_treatment_multi_relations(self):
        # https://nlp.johnsnowlabs.com/2022/06/28/drug_action_treatment_mapper_en_3_0.html

        text = """The patient is a 71-year-old female patient of Dr. X. and she was given Aklis, Dermovate, Aacidexam and Paracetamol."""
        p = nlu.load('en.med_ner.posology.small en.map_entity.drug_to_action_treatment')
        res = p.predict(text)
        for c in res.columns:
            print(res[c])


    def test_chunk_mapper_for_action_treatment_multi_relations(self):
        # https://nlp.johnsnowlabs.com/2022/06/28/drug_action_treatment_mapper_en_3_0.html

        text = """The patient is a 71-year-old female patient of Dr. X. and she was given Aklis, Dermovate, Aacidexam and Paracetamol."""
        p = nlu.load('en.med_ner.posology.small en.map_entity.drug_to_action_treatment en.map_entity.section_headers_normalized')
        res = p.predict(text)
        for c in res.columns:
             print(res[c])

    def test_chunk_mapper_with_resolver(self):
        # TODO need feature subsitution logic or inheritance
        pass
        # data = ['Sinequan 150 MG', 'Zonalon 50 mg']
        # p = nlu.load('en.resolve.rxnorm.augmented_re en.map_entity.rxnorm_to_action_treatment')
        # res = p.predict(data)
        # for c in res.columns:
        #     print(res[c])

    def test_all_chunk_mapper_pipes(self):
        text = """The patient is a 71-year-old female patient of Dr. X. and she was given Aklis, Dermovate, Aacidexam and Paracetamol."""

        tests = [

            'en.icd10cm_to_snomed',
            'en.icd10cm_to_umls',
            'en.icdo_to_snomed',
            'en.mesh_to_umls',
            'en.rxnorm_to_ndc',
            'en.rxnorm_to_umls',
            'en.snomed_to_icd10cm',
            'en.snomed_to_icdo',
            'en.snomed_to_umls',

        ]
        for t in tests:
            p = nlu.load(t,verbose=True)
            res = p.predict(text)
            for c in res.columns:
                print(res[c])

    def test_chunk_mapper_all(self):
        text = """The patient is a 71-year-old female patient of Dr. X. and she was given Aklis, Dermovate, Aacidexam and Paracetamol."""

        tests = ['en.map_entity.section_headers_normalized',
        'en.map_entity.abbreviation_to_definition',
        'en.map_entity.drug_to_action_treatment',
        'en.map_entity.drug_brand_to_ndc',
        'en.map_entity.icd10cm_to_snomed',
        'en.map_entity.icd10cm_to_umls',
        'en.map_entity.icdo_to_snomed',
        'en.map_entity.mesh_to_umls',
        'en.map_entity.rxnorm_to_action_treatment',
        'en.map_entity.rxnorm_resolver',
        'en.map_entity.rxnorm_to_ndc',
        'en.map_entity.rxnorm_to_umls',
        'en.map_entity.snomed_to_icd10cm',
        'en.map_entity.snomed_to_icdo',
        'en.map_entity.snomed_to_umls',]
        for t in tests:
            p = nlu.load('en.med_ner.posology.small ' + t)
            res = p.predict(text)
            for c in res.columns:
                print(res[c])


if __name__ == '__main__':
    unittest.main()

