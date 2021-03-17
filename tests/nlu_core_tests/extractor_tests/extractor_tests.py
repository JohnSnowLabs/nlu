

import unittest
import pandas as pd
import numpy as np
from nlu.extractors.extraction_resolver import OC_anno2config
from nlu.extractors.extraction_resolver_HC import HC_anno2config

from nlu.extractors.extractor_configs import  *
import nlu
from nlu.extractors.extractor_methods.base_extractor_methods import *
from nlu.extractors.extractor_methods.helper_extractor_methods import *
class TestExtractors(unittest.TestCase):
    def test_extractors_basic(self):
        # p = nlu.load('ner')
        # df = p.predict('I love America!', return_spark_df=True).toPandas()

        SPARK_NLP_LICENSE     = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJleHAiOjE2NDIyNzM5ODksImlhdCI6MTYxMDczNzk4OSwidW5pcXVlX2lkIjoiYjQyZTc3ODgtNTc2NS0xMWViLTgyMDgtNzIyM2RkN2MyNzY0In0.vXrrwHdBssA7X1D7yfved7nKzvpxvKAOOKViNBS19S_DBoPRyBX1AwoQaisi-3Wp3MFnHZNKl6EVPLb3xt4UXLDjWs_5Nr6l32DAx1VuEZCAvtGqAJZeJsV7cgrRrf3Gh8WM2XutZRgsqQn21pNNGDcmxLH_-4LfPOqzrL5nNbZ2RXT_U3mD6umK38nD6gHaOCDn_zbZsum3SSZ0yUybA8OaCFTE8nPv-fdREBYHmM3mKYwmHguJxJcQTkSEfayMDnqx2G6k90ZOo4LcblC9wHPigF3WtsRpsRd1s2DEDu8r9rqmqK2Uxl1bHl38KgIQBhE0Z26qUTW8Hg031QUFOA'
        AWS_ACCESS_KEY_ID     = 'AKIASRWSDKBGNV275I6B'
        AWS_SECRET_ACCESS_KEY = '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'
        JSL_SECRET            = '2.7.3-3f5059a2258ea6585a0bd745ca84dac427bca70c'

        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        import sparknlp_jsl
        ## TODO ASSERT may have different EMBED requirements than the NER model!
        p  = nlu.load('en.ner.ade.clinical en.assert', verbose=True)
        df = p.predict(['The patient has cancer and high fever and will die next week.', ' She had a seizure.'], drop_irrelevant_cols=False, metadata=True, return_spark_df=True).toPandas()
        df = p.predict(['The patient has cancer and high fever and will die next week.', ' She had a seizure.'])

    # # Extract Pyspark rows to list of Spark NLP Annotation dicts
    #     unpack_df = df.applymap(extract_pyspark_rows)
    #     ex_resolver = {}
    #     for c in p.components:
    #         print(c.model)
    #         if type(c.model) in OC_anno2config.keys():
    #             if OC_anno2config[type(c.model)]['default'] == '' :
    #                 print(f'COULD NOT FIND DEFAULT CONFIGS, USING FULL DEFAULT FOR MODEL ={c.model}')
    #                 ex_resolver[c.component_info.spark_output_column_names[0]] = OC_anno2config[type(c.model)]['default_full'](output_col_prefix=c.component_info.name)
    #             else :
    #                 ex_resolver[c.component_info.spark_output_column_names[0]] = OC_anno2config[type(c.model)]['default'](output_col_prefix=c.component_info.name)
    #         else:
    #             if HC_anno2config[type(c.model)]['default'] == '' :
    #                 print(f'COULD NOT FIND DEFAULT CONFIGS IN HC RESOLVER SPACE, USING FULL DEFAULT FOR MODEL ={c.model}')
    #                 ex_resolver[c.component_info.spark_output_column_names[0]] = HC_anno2config[type(c.model)]['default_full'](output_col_prefix=c.component_info.name)
    #             else :
    #                 ex_resolver[c.component_info.spark_output_column_names[0]] = HC_anno2config[type(c.model)]['default'](output_col_prefix=c.component_info.name)
    #
    #
    #
    #     merged_extraction_df = apply_extractors_and_merge(unpack_df,ex_resolver)
    #     print(df.columns)

        for c in df.columns:
            print(c)
            print(df[c])



    
if __name__ == '__main__':
    TestExtractors().test_entities_config()
    #     # MAP COL2EXTRACTOR CONFIG!
    #     'document'               : default_document_config(),
    #     # 'token'                  : default_tokenizer_config(),
    #     # 'pos'                    : default_POS_config(), # TODO
    #     # 'ner'                    : default_NER_config(),
    #     # 'word_embeddings'        : default_word_embedding_config(),
    #     # # 'language'               : default_language_classifier_config(),
    #     # 'sentence'               : default_sentence_detector_DL_config(),
    #     # 'chunk'                  : default_chunker_config(),
    #     # 'sentence_embeddings'    : default_sentence_embedding_config(),
    #     # 'chunk_embeddings'       : default_chunk_embedding_config(),
    #     # 'ner_chunk'              : default_ner_converter_config(),
    #     # 'T5'                     : default_T5_config(),
    #     # 'multi_category'         : default_multi_label_classifier_config('multi_cat'),
    #     # 'multi_category'         : default_full_config('multi_cat'),
    #     'stem'                   : default_stopwords_config(),#('stem'),
    #     'lemma'                  : default_lemma_config(),
    #     'cleanTokens'            : default_stopwords_config(),
    #
    #
    # }
