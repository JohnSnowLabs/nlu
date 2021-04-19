import unittest
import pandas as pd
import nlu
import tests.nlu_hc_tests.secrets as sct

class ChunkResolverTests(unittest.TestCase):

    def test_chunk_resolver(self):

        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET

        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        # res = nlu.load('en.ner.diseases en.resolve_chunk.snomed.findings', verbose=True).predict(['The patient has cancer and high fever and will die next week.', ' She had a seizure.'], drop_irrelevant_cols=False, metadata=True, )
        s1='The patient has COVID. He got very sick with it.'
        s2='Peter got the Corona Virus!'
        s3='COVID 21 has been diagnosed on the patient'
        s4 = """This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret's Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU"""
        s5 = "The patient has cancer and high fever and will die from Leukemia"
        text = [s1,s2,s3,s4,s5]
        #
        # by specifying output_level=chunk you will get one row per entity
        # https://nlp.johnsnowlabs.com/2021/02/04/redl_temporal_events_biobert_en.html
        data = """She is diagnosed with cancer in 1991.Then she was admitted to Mayo Clinic in May 2000 and discharged in October 2001"""
        # data = ["She is diagnosed with cancer in 1991.","Then she was admitted to Mayo Clinic in May 2000 and discharged in October 2001"]
        data = ["""DIAGNOSIS: Left breast adenocarcinoma stage T3 N1b M0, stage IIIA.
        She has been found more recently to have stage IV disease with metastatic deposits and recurrence involving the chest wall and lower left neck lymph nodes.
        PHYSICAL EXAMINATION
        NECK: On physical examination palpable lymphadenopathy is present in the left lower neck and supraclavicular area. No other cervical lymphadenopathy or supraclavicular lymphadenopathy is present.
        RESPIRATORY: Good air entry bilaterally. Examination of the chest wall reveals a small lesion where the chest wall recurrence was resected. No lumps, bumps or evidence of disease involving the right breast is present.
        ABDOMEN: Normal bowel sounds, no hepatomegaly. No tenderness on deep palpation. She has just started her last cycle of chemotherapy today, and she wishes to visit her daughter in Brooklyn, New York. After this she will return in approximately 3 to 4 weeks and begin her radiotherapy treatment at that time."""]
        data = ' Hello Peter how are you I like Angela Merkel from germany'
        # res= nlu.load('en.resolve_chunk.cpt_clinical').predict(data, output_level='chunk')
        # res= nlu.load('med_ner.jsl.wip.clinical en.resolve_chunk.cpt_clinical').predict(data, output_level='chunk')
        # data ="""The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion."""
        # res= nlu.load('med_ner.jsl.wip.clinical en.resolve_chunk.cpt_clinical').predict(data, output_level='chunk')
        #
        data = 'This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now. General: Well-developed female, in no acute distress, afebrile. HEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist. Neck: No lymphadenopathy. Chest: Clear. Abdomen: Positive bowel sounds and soft. Dermatologic: She has got redness along the lateral portion of her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short.'
        df = nlu.load('en.med_ner.ade.clinical').predict(data, output_level =  "chunk")

        print(res)
        for c in res.columns: print(res[c])
if __name__ == '__main__':
    ChunkResolverTests().test_entities_config()


