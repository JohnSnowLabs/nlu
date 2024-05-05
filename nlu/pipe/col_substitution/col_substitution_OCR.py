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

def substitute_document_classifier_text_cols(c, cols, is_unique=True, nlu_identifier=''):
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
            if 'visual_classifier_label.1' in cols:
                new_cols['visual_classifier_label.1'] = 'file_path'
            if 'visual_classifier_label' in cols:
                new_cols['visual_classifier_label'] = 'visual_classifier_prediction'

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
def substitute_document_classifier_text_cols(c, cols, is_unique=True, nlu_identifier=''):
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
        if 'visual_classifier_label.1' in cols:
            new_cols['visual_classifier_label.1'] = 'file_path'
        if 'visual_classifier_label' in cols:
            new_cols['visual_classifier_label'] = 'visual_classifier_prediction'

        new_cols[c] = c
    return new_cols  # TODO

def substitute_document_ner_cols(c, cols, nlu_identifier):
    """
    Drug Norm is always unique
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = 'entities' if nlu_identifier == 'UNIQUE' else f'entities_{nlu_identifier}'
    for c in cols:
        if '_ocr_confidence' in c:
            new_cols['meta_text_entity_confidence'] = f'{new_base_name}_confidence'
        if '_token' in c:
            new_cols['meta_text_entity_token'] = f'{new_base_name}_ner_entity'
        if '_entity_x' in c:
            new_cols['meta_text_entity_x'] = f'{new_base_name}_x_location'
        if '_entity_y' in c:
            new_cols['meta_text_entity_y'] = f'{new_base_name}_y_location'

        # new_cols[c] = c
    return new_cols

def substitute_form_extractor_text_cols(c, cols, is_unique=True, nlu_identifier=''):
    new_cols = {}
    for c in cols:
        if 'meta_visual_classifier_prediction_entity1' in c:
            new_cols['meta_visual_classifier_prediction_entity1'] = 'form_relation_prediction_key'
        if 'meta_visual_classifier_prediction_entity2' in c:
            new_cols['meta_visual_classifier_prediction_entity2'] = 'form_relation_prediction_value'
        # if 'path' in c:
        #     new_cols['path'] = 'file_path'
    return new_cols