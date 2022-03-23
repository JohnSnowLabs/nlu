"""Collection of methods to substitute cols of licensed component_to_resolve results"""
import logging

logger = logging.getLogger('nlu')


def substitute_recognized_text_cols(c, cols, is_unique=True, nlu_identifier=''):
    """
    Drug Norm is always unique
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    for c in cols:
        new_cols[c] = c
    return new_cols  # TODO
    # new_base_name = 'generic_classifier' if is_unique else f'generic_classification_{nlu_identifier}'
    # for col in cols :
    #     if '_results'    in col     : new_cols[col] = new_base_name
    #     elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
    #     elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
    #     elif '_embeddings' in col     : continue # irrelevant  new_cols[col] = f'{new_base_name}_embedding'
    #     elif '_types'      in col     : continue # new_cols[col] = f'{new_base_name}_type'
    #     elif 'meta' in col:
    #         if   '_sentence'       in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
    #         elif   'confidence'       in col  : new_cols[col] = f'{new_base_name}_confidence'  # maps to which sentence token comes from
    #
    #     else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    #     # new_cols[col]= f"{new_base_name}_confidence"
    # return new_cols
