

import tests.nlu_hc_tests.secrets as sct
import unittest
from nlu import *

class TestViz(unittest.TestCase):

    def test_viz_ner(self):
        pipe = nlu.load('ner.conll',verbose=True)
        data = "Donald Trump and Angela Merkel from Germany don't share many oppinions!"
        """If there are multiple components we chould VIZ from, we need to define what takes prescedence"""
        viz = pipe.viz(data,viz_type='ner')
        print(viz)
        print(viz)
        print(viz)


    def test_viz_dep(self):
        pipe = nlu.load('dep.typed',verbose=True)
        data = "Donald Trump and Angela Merkel from Germany don't share many oppinions!"
        viz = pipe.viz(data,viz_type='dep')
        print(viz)
        print(viz)
        print(viz)


    def test_viz_resolution_chunk(self):
        nlu_ref = 'en.resolve_chunk.icd10cm.neoplasms'
        data = """The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion."""

        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET

        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        pipe = nlu.load(nlu_ref,verbose=True)
        viz = pipe.viz(data,viz_type='resolution')
        print(viz)


    def test_viz_resolution_sentence(self):
        nlu_ref = 'en.resolve.icd10cm.augmented'
        data = "This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."
        secrets_json_path = '/home/ckl/old_home/Documents/freelance/jsl/tigerGraph/Presentation_2_graph_Summit/secrets.json'
        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        pipe = nlu.load(nlu_ref,verbose=True)
        viz = pipe.viz(data,viz_type='resolution')
        print(viz)
    def test_viz_relation(self):
        nlu_ref = 'med_ner.jsl.wip.clinical relation.temporal_events'
        data = "He was advised chest X-ray or CT scan after checking his SpO2 which was <= 93%"
        secrets_json_path = '/home/ckl/old_home/Documents/freelance/jsl/tigerGraph/Presentation_2_graph_Summit/secrets.json'
        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        pipe = nlu.load(nlu_ref,verbose=True)
        viz = pipe.viz(data,viz_type='relation')
        print(viz)




    def test_viz_assertion(self):
        nlu_ref = 'med_ner.jsl.wip.clinical assert'
        data = "The patient was tested for cancer, but none was detected, he is free of cancer."
        secrets_json_path = '/home/ckl/old_home/Documents/freelance/jsl/tigerGraph/Presentation_2_graph_Summit/secrets.json'
        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        pipe = nlu.load(nlu_ref,verbose=True)
        viz = pipe.viz(data,viz_type='assert')
        print(viz)


    def test_infer_viz_type(self):
        secrets_json_path = '/home/ckl/old_home/Documents/freelance/jsl/tigerGraph/Presentation_2_graph_Summit/secrets.json'
        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        nlu_ref = 'med_ner.jsl.wip.clinical assert'
        data = "The patient was tested for cancer, but none was detected, he is free of cancer."
        pipe = nlu.load(nlu_ref)
        pipe.viz(data)
if __name__ == '__main__':
    unittest.main()

