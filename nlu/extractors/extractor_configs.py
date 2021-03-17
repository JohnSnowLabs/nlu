"""
This file contains methods to get pre-defined configurations for every annotator.
Extractor_resolver.py should be used to resolve SparkNLP Annotator classes to methods
in this file, which return the corrosponding configs that need to be passed to
the master_extractor() call.

This file is where all the in extractor_base_data_classes.py Dataclasses are combined with the
extractors defined in helper_extractor_methods.py.


"""

from nlu.extractors.extractor_base_data_classes import SparkNLPExtractor,SparkNLPExtractorConfig
from nlu.extractors.extractor_methods.helper_extractor_methods import *
"""
This file contains methods to get pre-defined configurations for every annotator.
Extractor_resolver.py should be used to resolve SparkNLP Annotator classes to methods 
in this file, which return the corrosponding configs that need to be passed to 
the master_extractor() call.

This file is where all the in extractor_base_data_classes.py Dataclasses are combined with the 
extractors defined in extractor_methods.py.

"""
from nlu.extractors.extractor_methods.base_extractor_methods import *


def default_full_config(output_col_prefix='DEFAULT'):
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_positions       = True,
        get_begin           = True,
        get_end             = True,
        get_embeds          = True,
        get_result          = True,
        get_meta            = True,
        get_full_meta       = True,
        get_annotator_type  = True,
    )

def default_document_config(output_col_prefix='document'):
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
    )


def default_NER_config(output_col_prefix='NER'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        get_meta            = True,
        meta_white_list     = ['confidence'],
        name                = 'NER with IOB tags and confidences for them. ',
        description         = 'NER with IOB tags and confidences for them. ',
    )

def default_language_classifier_config(output_col_prefix='language'):
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        get_meta            = True,
        get_full_meta       = True,
        pop_result_list     = True,
        name                = 'Only keep maximum language confidence',
        description         = 'Instead of returning the confidence for every language the Classifier was traiend on, only the maximum confidence will be returned',
        meta_data_extractor = SparkNLPExtractor(meta_extract_language_classifier_max_confidence,
                                                'Extract the maximum confidence from all classified languages and drop the others. TODO top k results',
                                                'Keep only top language confidence')
    )


def default_only_result_config(output_col_prefix):
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        name                = 'Default result extractor',
        description         = 'Just get the result field'
    )
def default_only_embedding_config(output_col_prefix):
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_embeds          = True,
        name                = 'Default Embed extractor',
        description         = 'Just get the Embed field'
    )


def default_only_result_and_positions_config(output_col_prefix):
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        get_positions       = True,
        name                = 'Positional result only default',
        description         = 'Get the result field and the positions'
    )


def default_sentiment_dl_config(output_col_prefix='sentiment_dl'):
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        get_full_meta       = True,
        name                = 'Only keep maximum sentiment confidence ',
        description         = 'Instead of returning the confidence for Postive and Negative, only the confidence of the more likely class will be returned in the confidence column',
        meta_data_extractor = SparkNLPExtractor(meta_extract_maximum_binary_confidence,
                                                'Instead of returining positive/negative confidence, only the maximum confidence will be returned withouth sentence number reference.',
                                                'Maximum binary confidence')
    )

def default_sentiment_vivk_config(output_col_prefix='vivk_sentiment'):
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        get_full_meta       = True,
        name                = 'Default sentiment vivk',
        description         = 'Get prediction confidence and the resulting label'
    )




def default_tokenizer_config(output_col_prefix='token'):
    return default_only_result_config(output_col_prefix)

def default_POS_config(output_col_prefix='POS_tag'):
    return default_only_result_config(output_col_prefix)


def default_sentence_detector_DL_config(output_col_prefix='sentence'):
    return default_only_result_config(output_col_prefix)

def default_chunker_config(output_col_prefix='matched_chunk'):
    return default_only_result_config(output_col_prefix)


def default_ner_converter_config(output_col_prefix='ner_chunk'):
    return default_only_result_config(output_col_prefix)

# RLY TR%?
def default_T5_config(output_col_prefix='T5'):
    return default_only_result_config(output_col_prefix)



# EMBEDS
def default_sentence_embedding_config(output_col_prefix='sentence_embedding'):
    return default_only_embedding_config(output_col_prefix)

def default_chunk_embedding_config(output_col_prefix='chunk_embedding'):
    return default_only_embedding_config(output_col_prefix)

def default_word_embedding_config(output_col_prefix='word_embedding'):
    return default_only_embedding_config(output_col_prefix)

# TOKEN CLEANERS




def default_stopwords_config(output_col_prefix='stopwords_removed'):
    return default_only_result_config(output_col_prefix)
def default_lemma_config(output_col_prefix='lemma'):
    return default_only_result_config(output_col_prefix)
def default_stemm_config(output_col_prefix='stemm'):
    return default_only_result_config(output_col_prefix)

def default_sentence_detector_config(output_col_prefix='sentence'):
    return default_only_result_config(output_col_prefix)













#OLD BACKUP
#
#
#
#
#
# def meta_extract_language_classifier_max_confidence(row,configs):
#     ''' Extract the language classificationw ith highest confidence and drop the others '''
#     # Get the best, but what about TOP K! todo
#     # TODO conditional sentence extraction and mroe docs
#     #unpack all confidences to float and set 'sentence' key value to -1 so it does not affect finding the highest cnfidence
#     unpack_dict_values = lambda x : -1 if 'sentence' in x[0]  else float(x[1][0])
#     l = list(map(unpack_dict_values,row.items()))
#     m = np.argmax(l)
#     k = list(row.keys())[m]
#
#     return {k+'_confidence' : row[k][0]} # remoe [0] for list return
#
#
# def meta_extract_maximum_binary_confidence(row,configs):
#     ''' Extract the maximum confidence for a binary classifier that returns 2 confidences.
#     key schema is 'meta_' + configs.output_col_prefix + '_confidence'
#
#     Parameters
#     -------------
#     configs : SparkNLPExtractorConfig
#     if configs.get_sentence_origin is True, the sentence origin column will be kept, otherwise dropped.
#
#
#     row : dict
#         i.e. looks like {'meta_sentiment_dl_sentence': ['0', '1', '2'], 'meta_sentiment_dl_pos': ['1.0', '1.0', '1.0'], 'meta_sentiment_dl_neg': ['5.5978343E-11', '5.5978343E-11', '5.5978343E-11']}
#
#     Returns
#     ------------
#     dict
#       if configs.get_sentence_origin True  {'meta_sentiment_dl_sentence': ['0', '1'], 'meta_sentiment_dl_confidence': [0.9366506, 0.9366506]}
#       else {'meta_sentiment_dl_confidence': [0.9366506, 0.9366506]}
#     '''
#     # TODO inline these variables
#     meta_sent_key = 'meta_' + configs.output_col_prefix + '_sentence'
#     meta_conf_key_neg = 'meta_' + configs.output_col_prefix + '_neg'
#     meta_conf_key_pos = 'meta_' + configs.output_col_prefix + '_pos'
#     # Zip Pos/Neg conf column and keep max
#     keep_max = lambda x: max(float(x[0]), float(x[1]))
#     return {
#         **{'meta_' + configs.output_col_prefix + '_confidence':list(map(keep_max,zip(row[meta_conf_key_pos],row[meta_conf_key_neg])))},
#         **({'meta_' + configs.output_col_prefix + '_sentence' : row[meta_sent_key]} if configs.get_sentence_origin else {})
#     }
#
#
# def default_sentiment_dl_config(output_col_prefix='sentiment_dl'):
#     return SparkNLPExtractorConfig(
#         output_col_prefix   = output_col_prefix,
#         get_result          = True,
#         get_full_meta       = True,
#         name                = 'Only keep maximum sentiment confidence ',
#         description         = 'Instead of returning the confidence for Postive and Negative, only the confidence of the more likely class will be returned in the confidence column',
#         meta_data_extractor = SparkNLPExtractor(meta_extract_maximum_binary_confidence,
#                                                 'Instead of returining positive/negative confidence, only the maximum confidence will be returned withouth sentence number reference.',
#                                                 'Maximum binary confidence')
#     )
#
# def default_sentiment_vivk_config(output_col_prefix='vivk_sentiment'):
#     return SparkNLPExtractorConfig(
#         output_col_prefix   = output_col_prefix,
#         get_result          = True,
#         get_full_meta       = True,
#         name                = 'Default sentiment vivk',
#         description         = 'Get prediction confidence and the resulting label'
#     )