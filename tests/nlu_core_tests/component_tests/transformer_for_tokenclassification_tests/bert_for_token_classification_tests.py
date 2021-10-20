
import tests.nlu_hc_tests.secrets as sct

import unittest
from nlu import *
class BertForTokenClassificationTests(unittest.TestCase):
    """:cvar
                'en.embed.token_bert.conll03': 'bert_base_token_classifier_conll03',
            'en.embed.token_bert.large_conll03': 'bert_large_token_classifier_conll03',
            'en.embed.token_bert.ontonote': 'bert_base_token_classifier_ontonote',
            'en.embed.token_bert.large_ontonote': 'bert_large_token_classifier_ontonote',
            'en.embed.token_bert.few_nerd': 'bert_base_token_classifier_few_nerd',

               'en.embed.token_bert.classifier_ner_btc'	:'bert_token_classifier_ner_btc',
            'es.embed.token_bert.spanish_ner': 'bert_token_classifier_spanish_ner',
            'ja.embed.token_bert.classifier_ner_ud_gsd'	:'bert_token_classifier_ner_ud_gsd',

            'fa.embed.token_bert.parsbert_armanner': 'bert_token_classifier_parsbert_armanner',
            'fa.embed.token_bert.parsbert_ner': 'bert_token_classifier_parsbert_ner',
            'fa.embed.token_bert.parsbert_peymaner': 'bert_token_classifier_parsbert_peymaner',

            'sv.embed.token_bert.swedish_ner': 'bert_token_classifier_swedish_ner',

            'tr.embed.token_bert.turkish_ner': 'bert_token_classifier_turkish_ner',
            'tr.embed.token_bert.turkish_ner': 'bert_token_classifier_turkish_ner',

# HC LICENSED!
                'en.embed.token_bert.ner_clinical': 'bert_token_classifier_ner_clinical',
                'en.embed.token_bert.ner_jsl': 'bert_token_classifier_ner_jsl',
        'en.classify.token_bert.conll03',
         'en.classify.token_bert.large_conll03',
         'en.classify.token_bert.ontonote',
         'en.classify.token_bert.large_ontonote',
         'en.classify.token_bert.few_nerd',
         'es.classify.token_bert.spanish_ner', # TODO BUG???
         'ja.classify.token_bert.classifier_ner_ud_gsd',
         'fa.classify.token_bert.parsbert_armanner',
         'fa.classify.token_bert.parsbert_ner',
         'fa.classify.token_bert.parsbert_peymaner',
         'sv.classify.token_bert.swedish_ner',
         'tr.classify.token_bert.turkish_ner',








NLU 3.3.0 MODELS




    """

    def test_bert_for_token_classification(self):

        rf = [
            # # 'ig.embed.xlm_roberta'  ,
            # 'ig.embed_sentence.xlm_roberta' ,
            # 'lg.embed.xlm_roberta'  ,
            # 'lg.embed_sentence.xlm_roberta' ,
            # 'lou.embed.xlm_roberta' ,
            # 'pcm.embed.xlm_roberta' ,
            # 'pcm.embed_sentence.xlm_roberta' ,
            # 'wo.embed_sentence.xlm_roberta' ,
            # 'wo.embed.xlm_roberta' ,
            # 'rw.embed_sentence.xlm_roberta' ,
            # 'rw.embed.xlm_roberta'  ,
            # 'sw.embed_sentence.xlm_roberta' ,
            # 'sw.embed.xlm_roberta' ,
            # 'ha.embed.xlm_roberta'  ,
            # 'ha.embed_sentence.xlm_roberta' ,
            # 'am.embed.xlm_roberta'  ,
            # 'am.embed_sentence.xlm_roberta' , # OK !!!
            # 'yo.embed_sentence.xlm_roberta' ,
            # 'yo.embed.xlm_roberta' ,
            # 'fa.classify.token_roberta_token_classifier_zwnj_base_ner'	,
            # 'en.classify.token_roberta_large_token_classifier_conll03'	,
            # 'en.classify.token_roberta_base_token_classifier_ontonotes'	,
            # 'en.classify.token_roberta_base_token_classifier_conll03'	,
            'en.classify.token_distilroberta_base_token_classifier_ontonotes',
            'en.classify.token_albert_large_token_classifier_conll03'	,
            'en.classify.token_albert_base_token_classifier_conll03'	,
            'en.classify.token_xlnet_base_token_classifier_conll03'	,
            'en.classify.token_roberta.large_token_classifier_ontonotes' ,
            'en.classify.token_albert.xlarge_token_classifier_conll03'  ,
            'en.classify.token_xlnet.large_token_classifier_conll03' ,
            'en.classify.token_longformer.base_token_classifier_conll03' ,
            'xx.classify.token_xlm_roberta.token_classifier_ner_40_lang' ,
            'xx.embed.xlm_roberta_large'  ,

            'xx.classify.token_xlm_roberta.token_classifier_ner_40_lang', # TODO CRASH??
            # 'en.classify.token_xlnet.large_token_classifier_conll03',
            # 'en.classify.token_longformer.base_token_classifier_conll03',
        # Todo healthcare
        # 'en.classify.token_bert.ner_clinical',
        # 'en.classify.token_bert.ner_jsl',


        ]
        for r in rf :
            print(f"TESTING ref ==========================={r}")
            p = nlu.load(r)
            df = p.predict("Donald Trump and Angela Merkel are from America and Germany", metadata=True, output_level='token')
            for c in df.columns: print(df[c])

    def test_deidentification(self):
        SPARK_NLP_LICENSE     = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID     = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET            = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        rf = [
            # 'en.classify.token_bert.ner_deid', # TODO double check if all metadat we need is extracted ???
            # 'de.med_ner de.resolve.icd10gm  ',
            # 'de.med_ner de.resolve.snomed',

        ]
        for r in rf :
            print(f"TESTING ref ==========================={r}")
            p = nlu.load(r)
            df = p.predict("Donald Trump and Angela Merkel are from America and Germany", metadata=True, output_level='token')
            for c in df.columns: print(df[c])


if __name__ == '__main__':
    unittest.main()

