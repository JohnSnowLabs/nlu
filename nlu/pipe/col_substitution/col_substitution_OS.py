"""Collection of methods to substitute cols of open source component results
For most annotators we can re-use a common name.
Some make an exception and need special names derives, i.e. ClassifierDl/MultiClassifierDl/ etc.. which should be derived from nlu_ref.

"""
import logging
logger = logging.getLogger('nlu')

def substitute_base(c, cols, is_unique):
    """ Re-usable substitution method for, applicable to most annotators results.
    Exceptions are Annotators like ClassifierDl/MultiClassifierDl/ etc.. which should not be used with this method

    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    nlu_identifier = extract_nlu_identifier(c)
    new_base_name = '???TODO???' if is_unique else f'entities_{nlu_identifier}'
    for col in cols :
        if 'results'     in col     : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : new_cols[col] = f'{new_base_name}_embedding'
        elif 'meta' in col:
            if 'confidence' in col: new_cols[col]= f"{new_base_name}_confidence"
            elif 'entity' in     col: new_cols[col]= f"{new_base_name}_class"
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols



def substitute_ner_converter_cols(c, cols, is_unique):
    """
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = {}
    nlu_identifier = extract_nlu_identifier(c)
    new_base_name = 'entities' if is_unique else f'entities_{nlu_identifier}'
    for col in cols :
        if 'results'     in col     : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : new_cols[col] = f'{new_base_name}_embedding'
        elif 'meta' in col:
            if 'confidence' in col: new_cols[col]= f"{new_base_name}_confidence"
            elif 'entity' in     col: new_cols[col]= f"{new_base_name}_class"
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')

    return new_cols


def substitute_ner_dl_cols(c, cols, is_unique):
    """
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = []
    nlu_identifier =  extract_nlu_identifier(c)
    new_base_name = 'ner_iob' if is_unique else f'ner_iob_{nlu_identifier}'
    for col in cols :
        if 'results'     in col     : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col     : continue # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if 'confidence' in col: new_cols[col]= f"{new_base_name}_confidence"
            elif 'entity' in     col: new_cols[col]= f"{new_base_name}_class"
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
    return new_cols





def substitute_doc_assembler_cols(c, cols, is_unique=True):
    """
    Doc assember is always unique
    Fetched fields are:
    - entities@<storage_ref>_results
    - entities@<storage_ref>_<metadata>
        - entities@<storage_ref>_entity
        - entities@<storage_ref>_confidence
    """
    new_cols = []
    new_base_name = 'document'# if is_unique else f'document_{nlu_identifier}'
    for col in cols :
        if '_results'    in col     : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : new_cols[col] = f'{new_base_name}_embedding'
        elif '_types'      in col     : continue # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col: continue # This field is irrelevant, since meta_sentence in document assembler is always 0
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols

def substitute_sentence_detector_dl_cols(c, cols, is_unique=True):
    """
    Sent detector is always unique
    """
    new_cols = []
    new_base_name = 'sentence'# if is_unique else f'document_{nlu_identifier}'
    for col in cols :
        if '_results'    in col       : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : continue # Sentence never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col          : continue # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if 'sentence_sentence' in col: continue # Seems like an irrelevant field, so drop
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols



def substitute_tokenizer_cols(c, cols, is_unique=True):
    """
    Tokenizer is always unique
    """
    new_cols = []
    new_base_name = 'token'# if is_unique else f'document_{nlu_identifier}'
    for col in cols :
        if '_results'    in col       : new_cols[col] = new_base_name
        elif '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        elif '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        elif '_embeddings' in col     : continue # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        elif '_types' in col          : continue # new_cols[col] = f'{new_base_name}_type'
        elif 'meta' in col:
            if '_sentence' in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols



def substitute_word_embed_cols(c, cols, is_unique=True):
    """
    Substitute col name for Word Embeddings. For Word_Embeddings, some name will be infered, and word_embedding_<name> will become the base name schema
    """
    new_cols = []
    c_name   = extract_nlu_identifier()
    new_base_name = f'word_embedding_{c_name}'# if is_unique else f'document_{nlu_identifier}'
    for col in cols :
        if '_results'    in col     :# new_cols[col] = new_base_name can be omitted for word_embeddings, maps to the origin token, which will be in the tokenizer col anyways
        if '_beginnings' in col     : new_cols[col] = f'{new_base_name}_begin'
        if '_endings'    in col     : new_cols[col] = f'{new_base_name}_end'
        if '_embeddings' in col     : continue # Token never stores Embeddings  new_cols[col] = f'{new_base_name}_embedding'
        if '_types' in col          : continue # new_cols[col] = f'{new_base_name}_type'
        if 'meta' in col:
            if '_sentence' in col  : new_cols[col] = f'{new_base_name}_origin_sentence'  # maps to which sentence token comes from
            elif 'OOV'     in col  : new_cols[col] = f'{new_base_name}_is_OOV'  # maps to which sentence token comes from
            elif 'isWordStart' in col  : new_cols[col] = f'{new_base_name}_is_word_start'  # maps to which sentence token comes from
            elif 'pieceId' in col  : new_cols[col] = f'{new_base_name}_piece_id'  # maps to which sentence token comes from
            else : logger.info(f'Dropping unmatched metadata_col={col} for c={c}')
            # new_cols[col]= f"{new_base_name}_confidence"
    return new_cols





def extract_nlu_identifier(c):pass
