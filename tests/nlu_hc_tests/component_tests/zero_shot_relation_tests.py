import unittest

import nlu
import tests.secrets as sct


class ZeroShotRelationTests(unittest.TestCase):
    def test_zero_shot_relation_model(self):

        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )
        # b = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli','en','clinical/models')

        pipe = nlu.load('med_ner.clinical relation.zeroshot_biobert')
        # Configure relations to extract
        pipe['zero_shot_relation_extraction'].setRelationalCategories({
            "CURE": ["{{TREATMENT}} cures {{PROBLEM}}."],
            "IMPROVE": ["{{TREATMENT}} improves {{PROBLEM}}.", "{{TREATMENT}} cures {{PROBLEM}}."],
            "REVEAL": ["{{TEST}} reveals {{PROBLEM}}."]}).setMultiLabel(False)
        df = pipe.predict("Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer.")
        # res = nlu.load('en.ner.anatomy', verbose=True).predict(['The patient has cancer and a tumor and high fever and will die next week. He has pain in his left food and right upper brain', ' She had a seizure.'], drop_irrelevant_cols=False, metadata=True)
        print(df.columns)
        for c in df:
            print(c)
            print(df[c])

        print(df)


if __name__ == "__main__":
    ZeroShotRelationTests().test_zero_shot_relation_model()
