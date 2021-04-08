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
        name                = 'default_full',
        description         = 'Default full configuration, keeps all data and gets all metadata fields',

    )




def default_NER_converter_licensed_config(output_col_prefix='entities'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        get_full_meta       = True,
        meta_white_list     = ['entity','confidence'], #sentence, chunk
        name                = 'default_ner',
        description         = 'Converts IOB-NER representation into entity representation and generates confidences for the entire entity chunk',
    )

def default_chunk_resolution_config(output_col_prefix='resolved_entities'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        get_meta            = True,
        meta_white_list     = ['all_k_resolutions','all_k_distances', 'all_k_results','confidence','distance','target_text','all_k_aux_labels',
                               ], #sentence, chunk
        name                = 'default_ner',
        description         = 'Converts IOB-NER representation into entity representation and generates confidences for the entire entity chunk',
    )

def default_relation_extraction_positional_config(output_col_prefix='extracted_relations'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        meta_white_list     = [],
        get_meta            = True,
        meta_black_list     = ['entity1_begin','entity2_begin','entity1_end','entity2_end',],
        name                = 'default_relation_extraction',
        description         = 'Get relation extraction result and all metadata, positions of entities excluded',
    )


def positional_relation_extraction_config(output_col_prefix='extracted_relations'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_result          = True,
        get_meta            = True,
        get_full_meta       = True,
        name                = 'positional_relation_extraction',
        description         = 'Get relation extraction result and all metadata, which will include positions of entities chunks',
    )
