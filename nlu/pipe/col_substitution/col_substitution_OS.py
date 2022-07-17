"""Collection of methods to substitute cols of open source component_to_resolve results
For most annotators we can re-use a common name.
Some make an exception and need special names derives, i.e. ClassifierDl/MultiClassifierDl/ etc.. which should be derived from nlu_ref.

"""
import logging

logger = logging.getLogger('nlu')


def partially_implemented_substitutor(c, cols, nlu_identifier):
    """
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = nlu_identifier
    for col in cols:
        if 'results' in col:
            new_cols[col] = f'{new_base_name}_result'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif 'meta' in col:
            if 'confidence' in col:
                new_cols[col] = f"{new_base_name}_confidence"
            elif 'entity' in col:
                new_cols[col] = f"{new_base_name}_class"
            elif 'chunk' in col:
                new_cols[col] = f"{new_base_name}_origin_chunk"
            elif 'sentence' in col:
                new_cols[col] = f"{new_base_name}_origin_sentence"
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
                continue
        if '_embeddings' in col and f'{new_base_name}_embedding' not in new_cols.values():
            new_cols[col] = f'{new_base_name}_embedding'

    return new_cols


def substitute_ner_converter_cols(c, cols, nlu_identifier):
    """
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = 'entities' if nlu_identifier == 'UNIQUE' else f'entities_{nlu_identifier}' if 'entities' not in nlu_identifier else nlu_identifier
    for col in cols:
        if 'results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            new_cols[col] = f'{new_base_name}_embedding'
        elif 'meta' in col:
            if 'confidence' in col:
                new_cols[col] = f"{new_base_name}_confidence"
            elif 'entity' in col:
                new_cols[col] = f"{new_base_name}_class"
            elif 'sentence' in col:
                new_cols[col] = f"{new_base_name}_origin_sentence"
            elif 'chunk' in col:
                new_cols[col] = f"{new_base_name}_origin_chunk"
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_ner_dl_cols(c, cols, nlu_identifier):
    """
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = 'ner_iob' if nlu_identifier == 'UNIQUE' else f'ner_iob_{nlu_identifier}'
    for col in cols:
        if 'results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # always empty and irrelevant new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if 'confidence' in col:
                new_cols[col] = f"{new_base_name}_confidence"
            elif 'word' in col:
                continue  # is the same as token col, can be omitted
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_doc_assembler_cols(c, cols, nlu_identifier=True):
    """
    Doc assember is always unique
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = 'document'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # irrelevant  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                continue  # This field is irrelevant, since meta_sentence in document assembler is always 0
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_sentence_detector_dl_cols(c, cols, nlu_identifier=True):
    """
    Sent detector is always unique
    """
    new_cols = {}
    new_base_name = 'sentence' if nlu_identifier == 'UNIQUE' else f'sentence_dl'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Sentence never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if 'sentence_sentence' in col:
                continue  # Seems like an irrelevant field, so drop
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_sentence_detector_pragmatic_cols(c, cols, nlu_identifier=True):
    """
    Sent detector is always unique
    """
    new_cols = {}
    new_base_name = 'sentence' if nlu_identifier == 'UNIQUE' else f'sentence_pragmatic'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Sentence never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if 'sentence_sentence' in col:
                continue  # Seems like an irrelevant field, so drop
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_tokenizer_cols(c, cols, nlu_identifier=True):
    """
    Tokenizer is always unique
    """
    new_cols = {}
    new_base_name = 'token'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_transformer_token_classifier_cols(c, cols, nlu_identifier=True):
    """
    Token classifier
    """
    new_cols = {}
    new_base_name = 'classified_token' if nlu_identifier == 'UNIQUE' else f'classified_token_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col and 'Some' not in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col: new_cols[
                col] = f'{new_base_name}_origin_sentence'
            if 'Some' in col:
                new_cols[
                    col] = f"'{new_base_name}_{col.split('Some(')[-1].split(')')[0]}_confidence"
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_coref_cols(c, cols, nlu_identifier=True):
    """
    |ORIGIN_REFERENCE | CO_REFERENCES|
    | Peter           | he , him, that dude |
    | Maria           | her, she, the lady |

    # {'head': 'ROOT', 'head.end': '-1', 'sentence': '0', 'head.sentence': '-1', 'head.begin': '-1'}

    """
    new_cols = {}
    new_base_name = 'coref' if nlu_identifier == 'UNIQUE' else f'coref_{nlu_identifier}'
    for col in cols:
        if '_result' in col:
            new_cols[col] = new_base_name
        elif '_sentence' in col:
            new_cols[col] = f'{new_base_name}_origin_sentence'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col and 'Some' not in col:
            continue
        elif '_types' in col:
            continue
        elif 'meta' in col:
            if 'head.sentence' in col:
                new_cols[col] = f'{new_base_name}_head_origin_sentence'
            elif 'head.end' in col:
                new_cols[col] = f'{new_base_name}_head_end'
            elif 'head.begin' in col:
                new_cols[col] = f'{new_base_name}_head_begin'
            elif '_head' in col:
                new_cols[col] = f'{new_base_name}_head'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_seq_bert_classifier_cols(c, cols, nlu_identifier=True):
    """
    Seq classifier
    """
    new_cols = {}

    # Remove sequence from nlu_identifier because we use it in the base name anyways
    if '_sequence' in nlu_identifier:
        nlu_identifier = nlu_identifier.replace('_sequence', '')
    new_base_name = 'classified_sequence' if nlu_identifier == 'UNIQUE' else f'classified_sequence_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col and 'Some' not in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            if '_confidence' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            if 'Some' in col:
                new_cols[col] = f"'{new_base_name}_{col.split('Some(')[-1].split(')')[0]}_confidence"
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols


def substitute_span_classifier_cols(c, cols, nlu_identifier=True):
    """
    Seq classifier
    """
    new_cols = {}
    # Remove sequence from nlu_identifier because we use it in the base name anyways
    # if '_sequence' in nlu_identifier:
    #     nlu_identifier = nlu_identifier.replace('_sequence', '')
    new_base_name = 'answer' if nlu_identifier == 'UNIQUE' else f'{nlu_identifier}_answer'
    for col in cols:
        if 'span_result' in col:
            new_cols[col] = f'{new_base_name}'
        if 'span_score' in col:
            new_cols[col] = f'{new_base_name}_confidence'

        elif 'span_start_score' in col:
            new_cols[col] = f'{new_base_name}_start_confidence'
        elif 'span_end_score' in col:
            new_cols[col] = f'{new_base_name}_end_confidence'
        elif 'start' in col and not 'score' in col:
            new_cols[col] = f'{new_base_name}_start'
        elif 'end' in col and not 'score' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif 'sentence' in col:
            new_cols[col] = f'{new_base_name}_sentence'

    return new_cols


def substitute_multi_doc_span_assembler_cols(c, cols, nlu_identifier=True):
    """
    Seq classifier
    """
    new_cols = {}
    # Remove sequence from nlu_identifier because we use it in the base name anyways
    for col in cols:
        if 'question' in col and not 'context' in col:
            new_cols[col] = f'question'
        elif 'context' in col:
            new_cols[col] = f'context'
    return new_cols


def substitute_word_embed_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Word Embeddings. For Word_Embeddings, some name will be infered, and word_embedding_<name> will become the base name schema
    """
    new_cols = {}
    new_base_name = f'word_embedding_{nlu_identifier}' if 'word_embedding_' not in nlu_identifier else nlu_identifier  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            continue  # new_cols[col] = new_base_name can be omitted for word_embeddings, maps to the origin token, which will be in the tokenizer col anyways
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif 'OOV' in col:
                new_cols[col] = f'{new_base_name}_is_OOV'
            elif 'isWordStart' in col:
                new_cols[col] = f'{new_base_name}_is_word_start'
            elif 'pieceId' in col:
                new_cols[col] = f'{new_base_name}_piece_id'
            elif '_token' in col:
                continue  # Can be omitted, is the same as _result, just maps to origin_token
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
        elif '_embeddings' in col:
            new_cols[col] = new_base_name  # stores the embeds and represents basically the main result

    return new_cols


def substitute_sent_embed_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Word Embeddings. For Word_Embeddings, some name will be infered, and word_embedding_<name> will become the base name schema
    """
    new_cols = {}
    # new_base_name = f'sentence_embedding_{nlu_identifier}'  # if nlu_identifier else f'document_{nlu_identifier}'
    new_base_name = f'sentence_embedding_{nlu_identifier}' if 'sentence_embedding' not in nlu_identifier else nlu_identifier  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            continue  # new_cols[col] = new_base_name can be omitted for word_embeddings, maps to the origin token, which will be in the tokenizer col anyways
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if 'OOV' in col:
                new_cols[col] = f'{new_base_name}_is_OOV'
            elif 'isWordStart' in col:
                new_cols[col] = f'{new_base_name}_is_word_start'
            elif 'pieceId' in col:
                new_cols[col] = f'{new_base_name}_piece_id'
            elif '_token' in col:
                continue  # Can be omited, is the same as _result, just maps to origin_token
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
        elif '_embeddings' in col:
            new_cols[col] = new_base_name  # stores the embeds and represents basically the main result
        elif '_sentence' in col and 'meta' in col:
            new_cols[col] = f'{new_base_name}_origin_sentence'

    return new_cols


def substitute_chunk_embed_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for chunk Embeddings. For Word_Embeddings, some name will be infered, and chunk_embedding_<name> will become the base name schema
    """
    new_cols = {}
    new_base_name = f'chunk_embedding_{nlu_identifier}' if 'chunk_embedding_' not in nlu_identifier else nlu_identifier  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            continue  # new_cols[col] = new_base_name can be omitted for chunk_embeddings, maps to the origin chunk, which will be in the tokenizer col anyways
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif 'isWordStart' in col:
                new_cols[col] = f'{new_base_name}_is_word_start'
            elif 'pieceId' in col:
                new_cols[col] = f'{new_base_name}_piece_id'
            elif '_token' in col:
                continue  # Can be omited, is the same as _result, just maps to origin_token
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
        elif '_embeddings' in col:
            new_cols[col] = new_base_name  # stores the embeds and represents basically the main result

    return new_cols


def substitute_classifier_dl_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Word Embeddings. For Word_Embeddings, some name will be infered, and word_embedding_<name> will become the base name schema
    """
    new_cols = {}
    new_base_name = f'{nlu_identifier}'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[
                col] = new_base_name  # can be omitted for chunk_embeddings, maps to the origin chunk, which will be in the tokenizer col anyways
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            old_base_name = f'meta_{c.out_types[0]}'
            metadata = col.split(old_base_name)[-1]
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif metadata in ['confidence', '_confidence']:
                new_cols[col] = f'{new_base_name}_confidence'  # max confidence over all classes
            # else:
            #     new_cols[col] = f'{new_base_name}{metadata}_confidence'  # confidence field
            # else : logger.info(f'Dropping unmatched metadata_col={col} for os_components={os_components}')

    return new_cols


def substitute_ngram_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Word Embeddings. ngram will be the new base name
    """
    new_cols = {}
    new_base_name = f'ngram'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols


def substitute_labled_dependency_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Labled dependenecy labeled_dependency will become the base name schema
    """
    new_cols = {}
    new_base_name = f'labeled_dependency'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols


def substitute_un_labled_dependency_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Labled dependenecy unlabeled_dependency will become the base name schema
    """
    new_cols = {}
    new_base_name = f'unlabeled_dependency'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif '_head.begin' in col:
                new_cols[col] = f'{new_base_name}_head_begin'
            elif 'head.end' in col:
                new_cols[col] = f'{new_base_name}_head_end'
            elif 'head' in col:
                new_cols[col] = f'{new_base_name}_head'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols


def substitute_pos_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Labled dependenecy unlabeled_dependency will become the base name schema
    """
    new_cols = {}
    new_base_name = f'pos'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif '_word' in col:
                continue  # can be omitted, is jsut the token
            elif 'confidence' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_norm_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for normalized,  <norm> will be new base col name
    """
    new_cols = {}
    new_base_name = f'norm'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_doc_norm_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for normalized,  <norm> will be new base col name
    """
    new_cols = {}
    new_base_name = f'doc_norm'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_spell_context_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for normalized,  <spell> will be new base col namem
    1 spell checker is assumed per component_list for now
    """
    new_cols = {}
    new_base_name = f'spell' if nlu_identifier == 'UNIQUE' else f'spell_dl'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col: new_cols[col] = f'{new_base_name}_origin_sentence'
            if '_cost' in col:
                new_cols[col] = f'{new_base_name}_cost'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_spell_symm_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for sym,  <spell> will be new base col name
    1 spell checker is assumed per component_list for now
    """
    new_cols = {}
    new_base_name = f'spell' if nlu_identifier == 'UNIQUE' else f'spell_sym'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col: new_cols[col] = f'{new_base_name}_origin_sentence'
            if '_confidence' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_spell_norvig_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for spell,  <spell> will be new base col name
    1 spell checker is assumed per component_list for now
    """
    new_cols = {}
    new_base_name = f'spell' if nlu_identifier == 'UNIQUE' else f'spell_norvig'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col: new_cols[col] = f'{new_base_name}_origin_sentence'
            if '_confidence' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_word_seg_cols(c, cols, nlu_identifier=True):
    """
    Word_seg is always unique
    """
    new_cols = {}
    new_base_name = 'words_seg'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_stem_cols(c, cols, nlu_identifier=True):
    """
    stem is always unique
    """
    new_cols = {}
    new_base_name = 'stem'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_lem_cols(c, cols, nlu_identifier=True):
    """
    stem is always unique
    """
    new_cols = {}
    new_base_name = 'lem'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_stopwords_cols(c, cols, nlu_identifier=True):
    """
    stem is always unique
    """
    new_cols = {}
    new_base_name = 'stopword_less'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_chunk_cols(c, cols, nlu_identifier=True):
    """
    stem is always unique
    """
    new_cols = {}
    new_base_name = 'matched_pos'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_YAKE_cols(c, cols, nlu_identifier=True):
    """
    stem is always unique
    """
    new_cols = {}
    new_base_name = 'keywords'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col: new_cols[
                col] = f'{new_base_name}_origin_sentence'
            if '_score' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_marian_cols(c, cols, nlu_identifier=True):
    """
    rename cols with base name either <translated> or if not unique <translated_<lang>>
    """
    new_cols = {}
    new_base_name = 'translated' if nlu_identifier == 'UNIQUE' else f'translated_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_gpt2_cols(c, cols, nlu_identifier=True):
    """
    rename cols with base name either <gpt> or if not unique <generated_identifier>
    """
    new_cols = {}
    new_base_name = 'generated' if nlu_identifier == 'UNIQUE' else f'generated_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_T5_cols(c, cols, nlu_identifier=True):
    """
    rename cols with base name either <t5> or if not unique <t5_<task>>
    """
    new_cols = {}
    new_base_name = 't5' if nlu_identifier == 'UNIQUE' else f't5_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_sentiment_vivk_cols(c, cols, nlu_identifier=True):
    new_cols = {}
    new_base_name = 'sentiment' if nlu_identifier == 'UNIQUE' else f'sentiment_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif '_confidence' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_sentiment_dl_cols(c, cols, nlu_identifier=True):
    new_cols = {}
    new_base_name = 'sentiment' if nlu_identifier == 'UNIQUE' else f'sentiment_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif '_confidence' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            elif '_negative' in col:
                new_cols[col] = f'{new_base_name}_negative'
            elif '_positive' in col:
                new_cols[col] = f'{new_base_name}_positive'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_multi_classifier_dl_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Word Embeddings. For Word_Embeddings, some name will be infered, and word_embedding_<name> will become the base name schema
    """
    new_cols = {}
    new_base_name = f'{nlu_identifier}'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[
                col] = new_base_name  # can be omitted for chunk_embeddings, maps to the origin chunk, which will be in the tokenizer col anyways
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            old_base_name = f'meta_{c.out_types[0]}'
            metadata = col.split(old_base_name)[-1]
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif metadata == 'confidence':
                new_cols[col] = f'{new_base_name}_confidence'  # max confidence over all classes
            else:
                new_cols[col] = f'{new_base_name}{metadata}_confidence'  # confidence field
            # else : logger.info(f'Dropping unmatched metadata_col={col} for os_components={os_components}')

    return new_cols


def substitute_date_match_cols(c, cols, nlu_identifier=True):
    """
    stem is always unique
    """
    new_cols = {}
    new_base_name = 'matched_date'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_regex_match_cols(c, cols, nlu_identifier=True):
    """
    stem is always unique
    """
    new_cols = {}
    new_base_name = 'matched_regex'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_text_match_cols(c, cols, nlu_identifier=True):
    """
    stem is always unique
    """
    new_cols = {}
    new_base_name = 'matched_text'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


## Trainable
def substitute_classifier_dl_approach_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Word Embeddings. For Word_Embeddings, some name will be infered, and word_embedding_<name> will become the base name schema
    """
    new_cols = {}
    new_base_name = f'trained_classifier'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[
                col] = new_base_name  # can be omitted for chunk_embeddings, maps to the origin chunk, which will be in the tokenizer col anyways
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            old_base_name = f'meta_{c.out_types[0]}'
            metadata = col.split(old_base_name)[-1]
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif metadata == 'confidence':
                new_cols[col] = f'{new_base_name}_confidence'  # max confidence over all classes
            else:
                new_cols[col] = f'{new_base_name}{metadata}_confidence'  # confidence field
            # else : logger.info(f'Dropping unmatched metadata_col={col} for os_components={os_components}')
    return new_cols


def substitute_sentiment_vivk_approach_cols(c, cols, nlu_identifier=True):
    new_cols = {}
    new_base_name = 'trained_sentiment'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif '_confidence' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols


def substitute_sentiment_dl_approach_cols(c, cols, nlu_identifier=True):
    new_cols = {}
    new_base_name = 'trained_sentiment'
    for col in cols:
        if '_results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif '_confidence' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            elif '_negative' in col:
                new_cols[col] = f'{new_base_name}_negative'
            elif '_positive' in col:
                new_cols[col] = f'{new_base_name}_positive'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_multi_classifier_dl_approach_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Word Embeddings. For Word_Embeddings, some name will be infered, and word_embedding_<name> will become the base name schema
    """
    new_cols = {}
    new_base_name = f'trained_multi_classifier'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[
                col] = new_base_name  # can be omitted for chunk_embeddings, maps to the origin chunk, which will be in the tokenizer col anyways
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            old_base_name = f'meta_{c.out_types[0]}'
            metadata = col.split(old_base_name)[-1]
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif metadata == 'confidence':
                new_cols[col] = f'{new_base_name}_confidence'  # max confidence over all classes
            else:
                new_cols[col] = f'{new_base_name}{metadata}_confidence'  # confidence field
            # else : logger.info(f'Dropping unmatched metadata_col={col} for os_components={os_components}')

    return new_cols


def substitute_ner_dl_approach_cols(c, cols, nlu_identifier):
    """
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    new_base_name = 'trained_ner_iob'
    for col in cols:
        if 'results' in col:
            new_cols[col] = new_base_name
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col:
            continue  # always empty and irrelevant new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col:
            continue  # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if 'confidence' in col:
                new_cols[col] = f"{new_base_name}_confidence"
            elif 'word' in col:
                continue  # is the same as token col, can be omitted
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_pos_approach_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Labled dependenecy unlabeled_dependency will become the base name schema
    """
    new_cols = {}
    new_base_name = f'trained_pos'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            if '_sentence' in col:
                new_cols[col] = f'{new_base_name}_origin_sentence'
            elif '_word' in col:
                continue  # can be omitted, is jsut the token
            elif 'confidence' in col:
                new_cols[col] = f'{new_base_name}_confidence'
            else:
                logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols


def substitute_doc2chunk_cols(c, cols, nlu_identifier=True):
    """
    Substitute col name for Doc2chunk
    """
    new_cols = {}
    new_base_name = f'doc2chunk'  # if nlu_identifier else f'document_{nlu_identifier}'
    for col in cols:
        if '_results' in col:
            new_cols[col] = f'{new_base_name}'
        elif '_beginnings' in col:
            new_cols[col] = f'{new_base_name}_begin'
        elif '_endings' in col:
            new_cols[col] = f'{new_base_name}_end'
        elif '_types' in col:
            continue  #
        elif '_embeddings' in col:
            continue  #
        elif 'meta' in col:
            logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols
