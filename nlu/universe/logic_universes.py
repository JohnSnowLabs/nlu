"""
Collection of universes shared across all libraries (NLP/HC/OCR), which are collections of atoms
"""
from nlu.universe.atoms import LicenseType, NlpLevel


class NLP_LEVELS(NlpLevel):
    """:cvar
    XXX_SUPER is a N to M Mapping, with M <= N
    XXX_SUB is a N to M mapping, with M >=N
    no prefix implies a N to N mapping to be expected
    """
    DOCUMENT = NlpLevel('document')
    CHUNK = NlpLevel('chunk')
    NER_CHUNK = NlpLevel("ner_chunk")
    KEYWORD_CHUNK = NlpLevel('KEYWORD_CHUNK')
    POS_CHUNK = NlpLevel('POS_CHUNK')
    NGRAM_CHUNK = NlpLevel('NGRAM_CHUNK')
    SUB_CHUNK = NlpLevel('sub_chunk')
    SUPER_CHUNK = NlpLevel('super_chunk')
    SENTENCE = NlpLevel('sentence')
    RELATION = NlpLevel('relation')
    TOKEN = NlpLevel('token')
    SUB_TOKEN = NlpLevel('sub_token')
    SUPER_TOKEN = NlpLevel('super_token')

    INPUT_DEPENDENT_DOCUMENT_CLASSIFIER = NlpLevel('INPUT_DEPENDENT_DOCUMENT_CLASSIFIER')
    INPUT_DEPENDENT_DOCUMENT_EMBEDDING = NlpLevel('INPUT_DEPENDENT_DOCUMENT_EMBEDDING')
    # Same output level as the anno that generates the chunks
    INPUT_DEPENDENT_CHUNK_EMBEDDING = NlpLevel('INPUT_DEPENDENT_CHUNK_EMBEDDING')


class OCR_OUTPUT_LEVELS:
    # PAGES ARE LIKE TOKENS!! Book is full document!

    PAGES = 'pages'  # Generate 1 output per PAGE in each input document. I.e if 2 PDFs input with 5 pages each, gens 10 rows. 1 to many mapping
    FILE = 'file'  # Generate 1 output per document, I.e. 2 PDFS with 5 pages each gen 2 Row, 1 to one mapping
    OBJECT = 'object'  # Generate 1 output row per detected Object in Input document. I.e. if 2 PDFS with 5 Cats each, generates 10 rows. ---> REGION or Not?
    CHARACTER = 'character'  # Generate 1 oputput row per OCR'd character, I.e. 2 PDFS with 100 Chars each, gens 100 Rows.
    TABLE = 'table'  # 1 Pandas DF per Table.


class AnnoTypes:
    # DOCUMENT_XX can be sbustituted for SENTENCE
    TOKENIZER = 'tokenizer'
    TOKEN_CLASSIFIER = 'token_classifier'
    TRANSFORMER_TOKEN_CLASSIFIER = 'transformer_token_classifier'  # Can be token level but also NER level
    TRANSFORMER_SEQUENCE_CLASSIFIER = 'transformer_sequence_classifier'  # Can be token level but also NER level
    CHUNK_CLASSIFIER = 'chunk_classifier'  # ASSERTION/ NER GENERATES/CONTEXT_PARSER THESE but DOES NOT TAKE THEM IN!!! Split into NER-CHUNK Classifier, etc..?
    DOCUMENT_CLASSIFIER = 'document_classifier'
    RELATION_CLASSIFIER = 'relation_classifier'  # Pairs of chunks
    TOKEN_EMBEDDING = 'token_embedding'
    CHUNK_EMBEDDING = 'chunk_embedding'
    DOCUMENT_EMBEDDING = 'document_embedding'
    SENTENCE_DETECTOR = 'sentence_detector'
    SENTENCE_EMBEDDING = 'sentence_embedding'

    SPELL_CHECKER = 'spell_checker'
    HELPER_ANNO = 'helper_anno'
    TEXT_NORMALIZER = 'text_normalizer'
    TOKEN_NORMALIZER = 'token_normalizer'
    # TODO chunk sub-classes? I.e. POS-CHUNKS, NER-CHUNKS, KEYWORD-CHUNKS, RESOLUTION-CHUNKS, etc??
    pos_regex_chunker = 'token_normalizer'
    CHUNK_FILTERER = 'chunk_filterer'

    TEXT_RECOGNIZER = 'text_recognizer'