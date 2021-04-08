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
        get_meta            = True,
        meta_white_list     = ['entity','confidence'], #sentence, chunk
        name                = 'default_ner',
        description         = 'NER with IOB tags and confidences for them',
    )

