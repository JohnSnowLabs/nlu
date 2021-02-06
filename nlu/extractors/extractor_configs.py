from dataclasses import dataclass

@dataclass
class SparkNLPExtractorConfig:
    """
    Universal Configuration class for defining what data to extract from a Spark NLP annotator.
    These extractor configs can be passed to any extractor NLU defined for Spark-NLP.
    Setting a boolean config to false, results in the extractor NOT returning that field from the Annotator outputs
    """
    output_col_prefix     :str        # Prefix used for naming output columns
    get_begin             :bool       # Get Annotation beginnings
    get_end               :bool       # Get Annotation ends
    get_embeds            :bool       # Get Annotation Embeds
    get_result            :bool       # Get Annotation results
    get_meta              :bool       # get only relevant feature from meta map (Hard coded)
    get_full_meta         :bool       # get all keys and vals from base emta map
    get_annotator_type    :bool       # Get Annotator Type
    unpack_single_list    :bool       # Should unpack the result field. Only set true for annotators that return exactly one element in their result, like Document classifier! This will convert list with just 1 element into just their element in the final pandas representation
    meta_white_list       :List[str]  # Whitelist some keys which should be fetched from meta map
    meta_black_list       :List[str]  # black_list some keys which should not be fetched from meta map
    # { # dict of finisher extractor methods, which will be applied to specific fields aber finishing up base extraction
    #   'field' : extractor_method
    # }
def get_default_full_extractor_config(output_col_prefix='DEFAULT'):
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix,
        get_begin           = True,
        get_end             = True,
        get_embeds          = True,
        get_result          = True,
        get_meta            = True,
        get_full_meta       = True,
        meta_white_list = [],
        meta_black_list      = [],
        get_annotator_type  = True,
        unpack_single_list  = False
    )

def get_default_document_extractor_config():
    return SparkNLPExtractorConfig(
        output_col_prefix   = 'document',
        get_begin           = False,
        get_end             = False,
        get_embeds          = False,
        get_result          = True,
        get_meta            = False,
        get_full_meta       = False,
        meta_white_list = [],
        meta_black_list      = [],
        get_annotator_type  = False,
        unpack_single_list  = False
    )


def get_default_word_embedding_extractor_config():
    return SparkNLPExtractorConfig(
        output_col_prefix   = 'word_embedding',
        get_begin           = False,
        get_end             = False,
        get_embeds          = False,
        get_result          = True,
        get_meta            = False,
        get_full_meta       = False,
        meta_white_list = [],
        meta_black_list      = [],
        get_annotator_type  = False,
        unpack_single_list  = False
    )

def get_default_NER_extractor_config():
    return SparkNLPExtractorConfig(
        output_col_prefix   = 'NER',
        get_begin           = False,
        get_end             = False,
        get_embeds          = False,
        get_result          = True,
        get_meta            = True,
        get_full_meta       = False,
        meta_white_list     = ['confidence'],
        meta_black_list     = [],
        get_annotator_type  = False,
        unpack_single_list  = False
    )



def get_default_named_word_embedding_extractor_config(output_col_prefix='DEFAULT'):
    """Prefixes _word_embedding to input name"""
    return SparkNLPExtractorConfig(
        output_col_prefix   = output_col_prefix + '_word',
        get_begin           = False,
        get_end             = False,
        get_embeds          = False,
        get_result          = True,
        get_meta            = False,
        get_full_meta       = False,
        meta_black_list      = [],
        meta_white_list = [],
        get_annotator_type  = False,
        unpack_single_list  = False
    )