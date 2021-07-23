"""Collection of methods to substitute cols of licensed component results"""
import logging
logger = logging.getLogger('nlu')

def substitute_ner_internal_converter_cols(c, cols, nlu_identifier):
    """
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = 'entities' if nlu_identifier=='UNIQUE' else f'entities_{nlu_identifier}'
    for col in cols :
        if 'results'     in col     : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : new_cols[col] = f'{new_base_name}_embedding'
        elif 'meta' in col:
            if 'confidence' in col: new_cols[col]= f"{new_base_name}_confidence"
            elif 'entity' in     col: new_cols[col]= f"{new_base_name}_class"
            elif 'chunk' in     col: new_cols[col]= f"{new_base_name}_origin_chunk"
            elif 'sentence' in     col: new_cols[col]= f"{new_base_name}_origin_sentence"
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols





def substitute_chunk_resolution_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Resolution. For Resolution, some name will be infered, and entity_resolution_<name> will become the base name schema
all_k_results -> Sorted ResolverLabels in the top `alternatives` that match the distance `threshold`
all_k_resolutions -> Respective ResolverNormalized strings
all_k_distances -> Respective distance values after aggregation
all_k_wmd_distances -> Respective WMD distance values
all_k_tfidf_distances -> Respective TFIDF Cosinge distance values
all_k_jaccard_distances -> Respective Jaccard distance values
all_k_sorensen_distances -> Respective SorensenDice distance values
all_k_jaro_distances -> Respective JaroWinkler distance values
all_k_levenshtein_distances -> Respective Levenshtein distance values
all_k_confidences -> Respective normalized probabilities based in inverse distance values
target_text -> The actual searched string
resolved_text -> The top ResolverNormalized string
confidence -> Top probability
distance -> Top distance value
sentence -> Sentence index
chunk -> Chunk Index
token -> Token index
    """
    new_cols = {}
    new_base_name = f'entity_resolution' if nlu_identifier=='UNIQUE'  else f'entity_resolution_{nlu_identifier}'
    for col in cols :
        if '_results'      in col    and 'all_k' not in col :  new_cols[col] = f'{new_base_name}_code' # resolved code
        elif '_beginnings' in col     : new_cols[col]  = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col]  = f'{new_base_name}_end'
        elif '_types'      in col          : continue # new_cols[col] = f'{new_base_name}_type'
        elif '_embeddings' in col     : continue # omit , no data
        elif 'meta' in col:
            if   '_sentence' in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            elif 'all_k_aux_labels' in col  : new_cols[col] = f'{new_base_name}_k_aux_labels'  # maps to which sentence token comes from
            elif 'resolved_text' in col  : new_cols[col] = f'{new_base_name}' #The most likely resolution
            elif 'target_text' in col  : continue # Can be omitted, origin chunk basically, which will be included in the nerConverterInternal result
            elif 'all_k_confidences' in col  : new_cols[col] = f'{new_base_name}_k_confidences'  # confidences of the k resolutions
            elif 'confidence' in col  : new_cols[col] = f'{new_base_name}_confidence'
            elif 'all_k_results' in col  : new_cols[col] = f'{new_base_name}_k_results'
            elif 'all_k_distances' in col  : new_cols[col] = f'{new_base_name}_k_distances'
            elif 'all_k_resolutions' in col  : new_cols[col] = f'{new_base_name}_top_k'
            elif 'all_k_cosine_distances' in col  : new_cols[col] = f'{new_base_name}_k_cos_distances'
            elif 'all_k_wmd_distances' in col  : new_cols[col] = f'{new_base_name}_k_wmd_distances'
            elif 'all_k_tfidf_distances' in col  : new_cols[col] = f'{new_base_name}_k_tfidf_distances'
            elif 'all_k_jaccard_distances' in col  : new_cols[col] = f'{new_base_name}_k_jaccard_distances'
            elif 'all_k_sorensen_distances' in col  : new_cols[col] = f'{new_base_name}_k_sorensen_distances'
            elif 'all_k_jaro_distances' in col  : new_cols[col] = f'{new_base_name}_k_jaro_distances'
            elif 'all_k_levenshtein_distances' in col  : new_cols[col] = f'{new_base_name}_k_levenshtein_distances'
            elif 'distance' in col  : new_cols[col] = f'{new_base_name}_distance'
            elif 'chunk' in col  : continue # Omit, irreleant new_cols[col] = f'{new_base_name}_confidence'
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols


def substitute_sentence_resolution_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Resolution. For Resolution, some name will be infered, and sentence_resolution_<name> will become the base name schema
all_k_results -> Sorted ResolverLabels in the top `alternatives` that match the distance `threshold`
all_k_resolutions -> Respective ResolverNormalized strings
all_k_distances -> Respective distance values after aggregation
all_k_wmd_distances -> Respective WMD distance values
all_k_tfidf_distances -> Respective TFIDF Cosinge distance values
all_k_jaccard_distances -> Respective Jaccard distance values
all_k_sorensen_distances -> Respective SorensenDice distance values
all_k_jaro_distances -> Respective JaroWinkler distance values
all_k_levenshtein_distances -> Respective Levenshtein distance values
all_k_confidences -> Respective normalized probabilities based in inverse distance values
target_text -> The actual searched string
resolved_text -> The top ResolverNormalized string
confidence -> Top probability
distance -> Top distance value
sentence -> Sentence index
chunk -> Chunk Index
token -> Token index
    """
    new_cols = {}
    new_base_name = f'sentence_resolution' if nlu_identifier=='UNIQUE' else f'sentence_resolution_{nlu_identifier}'
    for col in cols :
        if '_results'      in col    and 'all_k' not in col :  new_cols[col] = f'{new_base_name}_code' # resolved code
        elif '_beginnings' in col     : new_cols[col]  = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col]  = f'{new_base_name}_end'
        elif '_types'      in col          : continue # new_cols[col] = f'{new_base_name}_type'
        elif '_embeddings' in col     : continue # omit , no data
        elif 'meta' in col:
            if 'all_k_aux_labels' in col  : new_cols[col] = f'{new_base_name}_k_aux_labels'  # maps to which sentence token comes from
            elif 'resolved_text' in col  : new_cols[col] = f'{new_base_name}' #The most likely resolution
            elif 'target_text' in col  : continue # Can be omitted, origin chunk basically, which will be included in the nerConverterInternal result
            elif 'all_k_confidences' in col  : new_cols[col] = f'{new_base_name}_k_confidences'  # confidences of the k resolutions
            elif 'confidence' in col  : new_cols[col] = f'{new_base_name}_confidence'
            elif 'all_k_results' in col  : new_cols[col] = f'{new_base_name}_k_results'
            elif 'all_k_distances' in col  : new_cols[col] = f'{new_base_name}_k_distances'
            elif 'all_k_resolutions' in col  : new_cols[col] = f'{new_base_name}_top_k'
            elif 'all_k_cosine_distances' in col  : new_cols[col] = f'{new_base_name}_k_cos_distances'
            elif 'all_k_wmd_distances' in col  : new_cols[col] = f'{new_base_name}_k_wmd_distances'
            elif 'all_k_tfidf_distances' in col  : new_cols[col] = f'{new_base_name}_k_tfidf_distances'
            elif 'all_k_jaccard_distances' in col  : new_cols[col] = f'{new_base_name}_k_jaccard_distances'
            elif 'all_k_sorensen_distances' in col  : new_cols[col] = f'{new_base_name}_k_sorensen_distances'
            elif 'all_k_jaro_distances' in col  : new_cols[col] = f'{new_base_name}_k_jaro_distances'
            elif 'all_k_levenshtein_distances' in col  : new_cols[col] = f'{new_base_name}_k_levenshtein_distances'
            elif 'distance' in col  : new_cols[col] = f'{new_base_name}_distance'
            elif 'chunk' in col  : continue # Omit, irreleant new_cols[col] = f'{new_base_name}_confidence'
            elif   '_sentence' in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols



def substitute_assertion_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Assertion. For Assertion, some name will be infered, and assertion_<sub_field> defines the base name schema
    Assert should always be unique
    """
    new_cols = {}
    # c_name   = extract_nlu_identifier(c)
    new_base_name = f'assertion'# if is_unique else f'sentence_resolution_{c_name}'
    for col in cols :
        if '_results'      in col     :  new_cols[col] = f'{new_base_name}' # resolved code
        elif '_beginnings' in col     : new_cols[col]  = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col]  = f'{new_base_name}_end'
        elif '_types'      in col     : continue # new_cols[col] = f'{new_base_name}_type'
        elif '_embeddings' in col     : continue # omit , no data
        elif 'meta' in col:
            if   '_sentence' in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            elif 'chunk' in col : new_cols[col] = f'{new_base_name}_origin_chunk'  # maps to which sentence token comes from
            elif 'confidence' in col  : new_cols[col] = f'{new_base_name}' #The most likely resolution
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols




def substitute_de_identification_cols(c, cols, is_unique=True):
    """
    Substitute col name for de-identification. For de-identification, some name will be infered, and de_identified_<sub_field> defines the base name schema
    de_identify should always be unique
    """
    new_cols = {}
    new_base_name = f'de_identified'#
    for col in cols :
        if '_results'      in col     :  new_cols[col] = f'{new_base_name}' # resolved code
        elif '_beginnings' in col     : new_cols[col]  = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col]  = f'{new_base_name}_end'
        elif '_types'      in col     : continue # new_cols[col] = f'{new_base_name}_type'
        elif '_embeddings' in col     : continue # omit , no data
        elif 'meta' in col:
            if   '_sentence' in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols



def substitute_relation_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for de-identification. For de-identification, some name will be infered, and de_identified_<sub_field> defines the base name schema
    de_identify should always be unique

              metadata = Map(
                "entity1" -> relation.entity1,
                "entity2" -> relation.entity2,
                "entity1_begin" -> relation.entity1_begin.toString,
                "entity1_end" -> relation.entity1_end.toString,
                "entity2_begin" -> relation.entity2_begin.toString,
                "entity2_end" -> relation.entity2_end.toString,
                "chunk1" -> relation.chunk1,
                "chunk2" -> relation.chunk2,
                "confidence" -> result._2.toString
              ),

    """
    new_cols = {}
    new_base_name = f'relation' if nlu_identifier=='UNIQUE' else f'relation_{nlu_identifier}'
    for col in cols :
        if '_results'      in col     : new_cols[col]  = f'{new_base_name}' # resolved code
        elif '_beginnings' in col     : new_cols[col]  = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col]  = f'{new_base_name}_end'
        elif '_types'      in col     : continue # new_cols[col] = f'{new_base_name}_type'
        elif '_embeddings' in col     : continue # omit , no data
        elif 'meta' in col:
            if   '_sentence'       in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            elif   'entity1_begin' in col  : new_cols[col] = f'{new_base_name}_entity1_begin'  # maps to which sentence token comes from
            elif   'entity2_begin' in col  : new_cols[col] = f'{new_base_name}_entity2_begin'  # maps to which sentence token comes from
            elif   'entity1_end'   in col  : new_cols[col] = f'{new_base_name}_entity1_end'  # maps to which sentence token comes from
            elif   'entity2_end'   in col  : new_cols[col] = f'{new_base_name}_entity2_end'  # maps to which sentence token comes from
            elif   'confidence'    in col  : new_cols[col] = f'{new_base_name}_confidence'  # maps to which sentence token comes from
            elif   'entity1'       in col  : new_cols[col] = f'{new_base_name}_entity1_class'  # maps to which sentence token comes from
            elif   'entity2'       in col  : new_cols[col] = f'{new_base_name}_entity2_class'  # maps to which sentence token comes from
            elif   'chunk1'        in col  : new_cols[col] = f'{new_base_name}_entity1'  # maps to which sentence token comes from
            elif   'chunk2'        in col  : new_cols[col] = f'{new_base_name}_entity2'  # maps to which sentence token comes from
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_drug_normalizer_cols(c, cols, is_unique=True):
    """
    Drug Norm is always unique
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = 'drug_norm'# if is_unique else f'document_{nlu_identifier}'
    for col in cols :
        if '_results'    in col     : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : continue # irrelevant  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types'      in col     : continue # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if   '_sentence'       in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols



def substitute_context_parser_cols(c, cols, is_unique=True):
    """
    Drug Norm is always unique
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = 'context_match'# if is_unique else f'document_{nlu_identifier}'
    for col in cols :
        if '_results'    in col     : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : continue # irrelevant  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types'      in col     : continue # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if   '_sentence'       in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            elif   'field'       in col  : new_cols[col] = f'{new_base_name}_field'  # maps to which sentence token comes from
            elif   'normalized'       in col  : new_cols[col] = f'{new_base_name}_normalized'  # maps to which sentence token comes from
            elif   'confidenceValue'       in col  : new_cols[col] = f'{new_base_name}_confidence'  # maps to which sentence token comes from
            elif   'hits'       in col  : new_cols[col] = f'{new_base_name}_hits'  # maps to which sentence token comes from

        else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_generic_classifier_parser_cols(c, cols, is_unique=True, nlu_identifier=''):
    """
    Drug Norm is always unique
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = 'generic_classifier' if is_unique else f'generic_classification_{nlu_identifier}'
    for col in cols :
        if '_results'    in col     : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : continue # irrelevant  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types'      in col     : continue # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if   '_sentence'       in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            elif   'confidence'       in col  : new_cols[col] = f'{new_base_name}_confidence'  # maps to which sentence token comes from

        else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
        # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


