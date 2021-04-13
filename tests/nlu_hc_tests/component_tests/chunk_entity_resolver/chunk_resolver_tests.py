import unittest
import pandas as pd
import nlu
import tests.nlu_hc_tests.secrets as sct

class ChunkResolverTests(unittest.TestCase):



    def test_quiczzzzzk(self):
        p_path = '/home/ckl/Downloads/tmp/analyze_sentiment_en_3.0.0_3.0_1616544471011'


        # res = nlu.load(path=p_path,verbose=True).predict('Am I the muppet or are you the muppet?')
        data = ['What is love?', 'Am Donald trump likes to party harty!I the muppet or are you the muppet?. I like to party party. Every day and night ','THis is the 3rd sent','this is 44.']

        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET

        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        # ref = 'en.classify.icd10.clinical'
        ref = p_path # 'en.classify.icd10.clinical'


        # res = nlu.load(path = f'{ref}' ,verbose=True).predict(data, output_level='document')
        # res = nlu.load(path = f'{ref}' ,verbose=True)#.predict(data, output_level='document')
        # res = nlu.load('en.med_ner.posology en.relation.drug_drug_interaction',verbose=True).predict(data, output_level='document')
        # res = nlu.load('en.med_ner.deid.large en.de_identify').predict('DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin on Friday 25. March 1999')
        s1='The patient has COVID. He got very sick with it.'
        s2='Peter got the Corona Virus!'
        s3='COVID 21 has been diagnosed on the patient'
        s4 = """This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret's Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU"""
        s5 = "The patient has cancer and high fever and will die from Leukemia"
        data = [s1,s2,s3,s4,s5]
        res = nlu.load('en.med_ner.posology en.relation.drug_drug_interaction').predict(data)
        # data = """The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature"""
        # data = """Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion. No right-sided pleural effusion or pneumothorax is definitively seen. There are mildly displaced fractures of the left lateral 8th and likely 9th ribs."""
        # res = nlu.load('en.med_ner.diseases en.resolve.snomed ', verbose=True).predict(data,output_level='chunk')
        res
        for c in res :print(res[c])

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
        data = [s1,s2,s3,s4,s5]

        res = nlu.load('en.resolve_sentence.hcc.augmented', verbose=True).predict(data, drop_irrelevant_cols=False, metadata=True, )

        print(res.columns)
        for c in res :
            print(res[c])

        print(res)
if __name__ == '__main__':
    ChunkResolverTests().test_entities_config()


