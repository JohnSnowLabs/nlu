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
        s0="""DIAGNOSIS: Left breast adenocarcinoma stage T3 N1b M0, stage IIIA. She has been found more recently to have stage IV disease with metastatic deposits and recurrence involving the chest wall and lower left neck lymph nodes. PHYSICAL EXAMINATION NECK: On physical examination palpable lymphadenopathy is present in the left lower neck and supraclavicular area. No other cervical lymphadenopathy or supraclavicular lymphadenopathy is present. RESPIRATORY: Good air entry bilaterally. Examination of the chest wall reveals a small lesion where the chest wall recurrence was resected. No lumps, bumps or evidence of disease involving the right breast is present. ABDOMEN: Normal bowel sounds, no hepatomegaly. No tenderness on deep palpation. She has just started her last cycle of chemotherapy today, and she wishes to visit her daughter in Brooklyn, New York. After this she will return in approximately 3 to 4 weeks and begin her radiotherapy treatment at that time."""
        s1='The patient has COVID. He got very sick with it.'
        s2='Peter got the Corona Virus!'
        s3='COVID 21 has been diagnosed on the patient'
        s4 = """This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret's Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU"""
        s5 = "The patient has cancer and high fever and will die from Leukemia"
        s6 = 'This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now. General: Well-developed female, in no acute distress, afebrile. HEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist. Neck: No lymphadenopathy. Chest: Clear. Abdomen: Positive bowel sounds and soft. Dermatologic: She has got redness along the lateral portion of her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short.'
        data = [s1,s2,s3,s4,s5,s6]
        res = nlu.load('med_ner.jsl.wip.clinical resolve_chunk.icdo.clinical', verbose=True).predict(data, drop_irrelevant_cols=False, metadata=True, )

        print(res.columns)
        for c in res.columns: print(res[c])
if __name__ == '__main__':
    ChunkResolverTests().test_entities_config()


"""

# COLS FOR MULTI SENTENCE WHERE SOME DONT HAVE ENTITIES OR SO , TODO BUGFIX!!@!!
IF SOME ROWS ARE MISSING SOME LABE<S THEN GGGGG FCK FUX
Index(['document_beginnings', 'document_endings', 'document_results',
       'document_types', 'document_embeddings', 'meta_document_sentence',
       'sentence_beginnings', 'sentence_endings', 'sentence_results',
       'sentence_types', 'sentence_embeddings', 'meta_sentence_sentence',
       'token_beginnings', 'token_endings', 'token_results', 'token_types',
       'token_embeddings', 'meta_token_sentence',
       'word_embeddings@clinical_beginnings',
       'word_embeddings@clinical_endings', 'word_embeddings@clinical_results',
       'word_embeddings@clinical_types', 'word_embeddings@clinical_embeddings',
       'meta_word_embeddings@clinical_sentence',
       'meta_word_embeddings@clinical_isOOV',
       'meta_word_embeddings@clinical_isWordStart',
       'meta_word_embeddings@clinical_pieceId',
       'meta_word_embeddings@clinical_token', 'ner@diseases_beginnings',
       'ner@diseases_endings', 'ner@diseases_results', 'ner@diseases_types',
       'ner@diseases_embeddings', 'meta_ner@diseases_word',
       'meta_ner@diseases_confidence', 'origin_index'],
      dtype='object')




# COLS FOR 1 SENTENCE
Index(['document_beginnings', 'document_endings', 'document_results',
       'document_types', 'document_embeddings', 'meta_document_sentence',
       'sentence_beginnings', 'sentence_endings', 'sentence_results',
       'sentence_types', 'sentence_embeddings', 'meta_sentence_sentence',
       'token_beginnings', 'token_endings', 'token_results', 'token_types',
       'token_embeddings', 'meta_token_sentence',
       'word_embeddings@clinical_beginnings',
       'word_embeddings@clinical_endings', 'word_embeddings@clinical_results',
       'word_embeddings@clinical_types', 'word_embeddings@clinical_embeddings',
       'meta_word_embeddings@clinical_sentence',
       'meta_word_embeddings@clinical_isOOV',
       'meta_word_embeddings@clinical_isWordStart',
       'meta_word_embeddings@clinical_pieceId',
       'meta_word_embeddings@clinical_token', 'ner@diseases_beginnings',
       'ner@diseases_endings', 'ner@diseases_results', 'ner@diseases_types',
       'ner@diseases_embeddings', 'meta_ner@diseases_word',
       'meta_ner@diseases_confidence', 'entities@diseases_beginnings',
       'entities@diseases_endings', 'entities@diseases_results',
       'entities@diseases_types', 'entities@diseases_embeddings',
       'meta_entities@diseases_sentence', 'meta_entities@diseases_chunk',
       'meta_entities@diseases_entity', 'meta_entities@diseases_confidence',
       'chunk_embeddings@clinical_beginnings',
       'chunk_embeddings@clinical_endings',
       'chunk_embeddings@clinical_results', 'chunk_embeddings@clinical_types',
       'chunk_embeddings@clinical_embeddings',
       'meta_chunk_embeddings@clinical_sentence',
       'meta_chunk_embeddings@clinical_chunk',
       'meta_chunk_embeddings@clinical_isWordStart',
       'meta_chunk_embeddings@clinical_pieceId',
       'meta_chunk_embeddings@clinical_token', 'chunk_resolution_beginnings',
       'chunk_resolution_endings', 'chunk_resolution_results',
       'chunk_resolution_types', 'chunk_resolution_embeddings',
       'meta_chunk_resolution_all_k_aux_labels',
       'meta_chunk_resolution_sentence', 'meta_chunk_resolution_resolved_text',
       'meta_chunk_resolution_target_text',
       'meta_chunk_resolution_all_k_confidences',
       'meta_chunk_resolution_distance', 'meta_chunk_resolution_confidence',
       'meta_chunk_resolution_chunk', 'meta_chunk_resolution_all_k_results',
       'meta_chunk_resolution_token', 'meta_chunk_resolution_all_k_distances',
       'meta_chunk_resolution_all_k_resolutions',
       'meta_chunk_resolution_all_k_cosine_distances', 'origin_index'],
      dtype='object')

"""