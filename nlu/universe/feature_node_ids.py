"""
Collection JSL Annotator IDs used by NLU internally
"""
from nlu.universe.atoms import JslAnnoId


class NLP_NODE_IDS:
    """All avaiable Feature nodes in NLP..
    Used to cast the pipeline dependency resolution algorithm into an abstract graph
    """
    # Visual Document Understanding
    BIG_TEXT_MATCHER = JslAnnoId('big_text_matcher')
    CHUNK2DOC = JslAnnoId('chunk2doc')
    CHUNK_EMBEDDINGS_CONVERTER = JslAnnoId('chunk_embeddings_converter')
    CHUNK_TOKENIZER = JslAnnoId('chunk_tokenizer')
    CHUNKER = JslAnnoId('chunker')
    CLASSIFIER_DL = JslAnnoId('classifier_dl')
    CONTEXT_SPELL_CHECKER = JslAnnoId('context_spell_checker')
    DATE_MATCHER = JslAnnoId('date_matcher')
    UNTYPED_DEPENDENCY_PARSER = JslAnnoId('untyped_dependency_parser')
    TYPED_DEPENDENCY_PARSER = JslAnnoId('typed_dependency_parser')
    DOC2CHUNK = JslAnnoId('doc2chunk')
    DOC2VEC = JslAnnoId('doc2vec')  # TODO ADD NODE!!
    TRAIANBLE_DOC2VEC = JslAnnoId('trainable_doc2vec')  # TODO ADD NODE!!
    DOCUMENT_ASSEMBLER = JslAnnoId('document_assembler')
    DOCUMENT_NORMALIZER = JslAnnoId('document_normalizer')
    EMBEDDINGS_FINISHER = JslAnnoId('embeddings_finisher')
    ENTITY_RULER = JslAnnoId('entitiy_ruler')
    FINISHER = JslAnnoId('FINISHER')
    GRAPH_EXTRACTION = JslAnnoId('graph_extraction')
    GRAPH_FINISHER = JslAnnoId('graph_finisher')
    LANGUAGE_DETECTOR_DL = JslAnnoId('language_detector_dl')
    LEMMATIZER = JslAnnoId('lemmatizer')
    MULTI_CLASSIFIER_DL = JslAnnoId('multi_classifier_dl')
    MULTI_DATE_MATCHER = JslAnnoId('multi_date_matcher')
    N_GRAMM_GENERATOR = JslAnnoId('n_gramm_generator')
    NER_CONVERTER = JslAnnoId('ner_converter')
    NER_CRF = JslAnnoId('ner_crf')
    NER_DL = JslAnnoId('ner_dl')
    NER_OVERWRITER = JslAnnoId('ner_overwriter')
    NORMALIZER = JslAnnoId('normalizer')
    NORVIG_SPELL_CHECKER = JslAnnoId('norvig_spell_checker')
    POS = JslAnnoId('pos')
    RECURISVE_TOKENIZER = JslAnnoId('recursive_tokenizer')
    REGEX_MATCHER = JslAnnoId('regex_matcher')
    REGEX_TOKENIZER = JslAnnoId('regex_tokenizer')
    SENTENCE_DETECTOR = JslAnnoId('sentence_detector')
    SENTENCE_DETECTOR_DL = JslAnnoId('sentence_detector_dl')
    SENTENCE_EMBEDDINGS_CONVERTER = JslAnnoId('sentence_embeddings_converter')
    STEMMER = JslAnnoId('stemmer')
    STOP_WORDS_CLEANER = JslAnnoId('stop_words_cleaner')
    SYMMETRIC_DELETE_SPELLCHECKER = JslAnnoId('symmetric_delete_spellchecker')
    TEXT_MATCHER = JslAnnoId('text_matcher')
    TOKEN2CHUNK = JslAnnoId('token2chunk')
    TOKEN_ASSEMBLER = JslAnnoId('token_assembler')
    TOKENIZER = JslAnnoId('tokenizer')
    SENTIMENT_DL = JslAnnoId('sentiment_dl')
    SENTIMENT_DETECTOR = JslAnnoId('sentiment_detector')
    VIVEKN_SENTIMENT = JslAnnoId('vivekn_sentiment')
    WORD_EMBEDDINGS = JslAnnoId('word_embeddings')
    WORD_SEGMENTER = JslAnnoId('word_segmenter')
    YAKE_KEYWORD_EXTRACTION = JslAnnoId('yake_keyword_extraction')
    ALBERT_EMBEDDINGS = JslAnnoId('albert_embeddings')
    ALBERT_FOR_TOKEN_CLASSIFICATION = JslAnnoId('albert_for_token_classification')
    BERT_EMBEDDINGS = JslAnnoId('bert_embeddings')
    BERT_FOR_TOKEN_CLASSIFICATION = JslAnnoId('bert_for_token_classification')
    BERT_SENTENCE_EMBEDDINGS = JslAnnoId('bert_sentence_embeddings')
    DISTIL_BERT_EMBEDDINGS = JslAnnoId('distil_bert_embeddings')
    DISTIL_BERT_FOR_TOKEN_CLASSIFICATION = JslAnnoId('distil_bert_for_token_classification')
    DISTIL_BERT_FOR_SEQUENCE_CLASSIFICATION = JslAnnoId('distil_bert_for_sequence_classification')
    BERT_FOR_SEQUENCE_CLASSIFICATION = JslAnnoId('bert_for_sequence_classification')
    ELMO_EMBEDDINGS = JslAnnoId('elmo_embeddings')
    LONGFORMER_EMBEDDINGS = JslAnnoId('longformer_embeddings')
    LONGFORMER_FOR_TOKEN_CLASSIFICATION = JslAnnoId('longformer_for_token_classification')
    MARIAN_TRANSFORMER = JslAnnoId('marian_transformer')
    ROBERTA_EMBEDDINGS = JslAnnoId('roberta_embeddings')
    ROBERTA_FOR_TOKEN_CLASSIFICATION = JslAnnoId('roberta_for_token_classification')
    ROBERTA_SENTENCE_EMBEDDINGS = JslAnnoId('roberta_sentence_embeddings')
    T5_TRANSFORMER = JslAnnoId('t5_transformer')
    UNIVERSAL_SENTENCE_ENCODER = JslAnnoId('universal_sentence_encoder')
    XLM_ROBERTA_EMBEDDINGS = JslAnnoId('xlm_roberta_embeddings')
    XLM_ROBERTA_FOR_TOKEN_CLASSIFICATION = JslAnnoId('xlm_roberta_for_token_classification')
    XLM_ROBERTA_SENTENCE_EMBEDDINGS = JslAnnoId('xlm_roberta_sentence_embeddings')
    XLNET_EMBEDDINGS = JslAnnoId('xlnet_embeddings')
    XLNET_FOR_TOKEN_CLASSIFICATION = JslAnnoId('xlnet_for_token_classification')
    XLM_ROBERTA_FOR_SEQUENCE_CLASSIFICATION = JslAnnoId('xlm_roberta_for_sequence_classification')
    ROBERTA_FOR_SEQUENCE_CLASSIFICATION = JslAnnoId('roberta_for_sequence_classification')
    LONGFORMER_FOR_SEQUENCE_CLASSIFICATION = JslAnnoId('longformer_for_sequence_classification')
    ALBERT_FOR_SEQUENCE_CLASSIFICATION = JslAnnoId('albert_for_sequence_classification')
    XLNET_FOR_SEQUENCE_CLASSIFICATION = JslAnnoId('xlnet_for_sequence_classification')
    GPT2 = JslAnnoId('gpt2')


    TRAINABLE_CONTEXT_SPELL_CHECKER = JslAnnoId('trainable_context_spell_checker')
    TRAINABLE_VIVEKN_SENTIMENT = JslAnnoId('trainable_vivekn_sentiment')
    TRAINABLE_SENTIMENT_DL = JslAnnoId('trainable_sentiment_dl')
    TRAINABLE_CLASSIFIER_DL = JslAnnoId('trainable_classifier_dl')
    TRAINABLE_MULTI_CLASSIFIER_DL = JslAnnoId('trainable_multi_classifier_dl')
    TRAINABLE_NER_DL = JslAnnoId('trainable_ner_dl')
    TRAINABLE_NER_CRF = JslAnnoId('trainable_ner_crf')
    TRAINABLE_POS = JslAnnoId('trainable_pos')
    TRAINABLE_DEP_PARSE_TYPED = JslAnnoId('trainable_dependency_parser')
    TRAINABLE_DEP_PARSE_UN_TYPED = JslAnnoId('trainable_dependency_parser_untyped')
    TRAINABLE_DOC2VEC = JslAnnoId('trainable_doc2vec')
    TRAINABLE_ENTITY_RULER = JslAnnoId('trainable_entity_ruler')
    TRAINABLE_LEMMATIZER = JslAnnoId('trainable_lemmatizer')
    TRAINABLE_NORMALIZER = JslAnnoId('trainable_normalizer')
    TRAINABLE_NORVIG_SPELL_CHECKER = JslAnnoId('trainable_norvig_spell')
    TRAINABLE_RECURISVE_TOKENIZER = JslAnnoId('trainable_recursive_tokenizer')
    TRAINABLE_REGEX_MATCHER = JslAnnoId('trainable_regex_tokenizer')
    TRAINABLE_SENTENCE_DETECTOR_DL = JslAnnoId('trainable_sentence_detector_dl')
    TRAINABLE_SENTIMENT = JslAnnoId('trainable_sentiment')
    TRAINABLE_WORD_EMBEDDINGS = JslAnnoId('trainable_word_embeddings')
    TRAINABLE_SYMMETRIC_DELETE_SPELLCHECKER = JslAnnoId('trainable_symmetric_spell_checker')
    TRAINABLE_TEXT_MATCHER = JslAnnoId('trainable_text_matcher')
    TRAINABLE_TOKENIZER = JslAnnoId('trainable_tokenizer')
    TRAINABLE_WORD_SEGMENTER = JslAnnoId('trainable_word_segmenter')


class NLP_HC_NODE_IDS:  # or Mode Node?
    """All avaiable Feature nodes in Healthcare Library.
    Defines High Level Identifiers

    Used to cast the pipeline dependency resolution algorithm into an abstract grpah
    """
    ASSERTION_DL = JslAnnoId('assertion_dl')
    TRAINABLE_ASSERTION_DL = JslAnnoId('trainable_assertion_dl')
    ASSERTION_FILTERER = JslAnnoId('assertion_filterer')  # TODO traianble?
    ASSERTION_LOG_REG = JslAnnoId('assertion_log_reg')
    TRAINABLE_ASSERTION_LOG_REG = JslAnnoId('trainable_assertion_log_reg')
    CHUNK2TOKEN = JslAnnoId('chunk2token')
    CHUNK_ENTITY_RESOLVER = JslAnnoId('chunk_entity_resolver')
    TRAINABLE_CHUNK_ENTITY_RESOLVER = JslAnnoId('traianble_chunk_entity_resolver')
    CHUNK_FILTERER = JslAnnoId('chunk_filterer')
    TRAINABLE_CHUNK_FILTERER = JslAnnoId('trainable_chunk_filterer')  # TODO feature node entires!!!
    CHUNK_KEY_PHRASE_EXTRACTION = JslAnnoId('chunk_key_phrase_extraction')
    CHUNK_MERGE = JslAnnoId('chunk_merge')
    TRAINABLE_CHUNK_MERGE = JslAnnoId('trainable_chunk_merge')  # TODO feature node entriess!!
    CONTEXTUAL_PARSER = JslAnnoId('contextual_parser')
    TRAIANBLE_CONTEXTUAL_PARSER = JslAnnoId('trainable_contextual_parser')  # TODO feature node entriess!!
    DE_IDENTIFICATION = JslAnnoId('de_identification')
    TRAINABLE_DE_IDENTIFICATION = JslAnnoId('trainable_de_identification')
    DOCUMENT_LOG_REG_CLASSIFIER = JslAnnoId('document_log_reg_classifier')
    TRAINABLE_DOCUMENT_LOG_REG_CLASSIFIER = JslAnnoId('traianble_document_log_reg_classifier')
    DRUG_NORMALIZER = JslAnnoId('drug_normalizer')
    FEATURES_ASSEMBLER = JslAnnoId('features_assembler')
    GENERIC_CLASSIFIER = JslAnnoId('generic_classifier')
    TRAINABLE_GENERIC_CLASSIFIER = JslAnnoId('traianble_generic_classifier')
    IOB_TAGGER = JslAnnoId('iob_tagger')
    MEDICAL_NER = JslAnnoId('medical_ner')
    TRAINABLE_MEDICAL_NER = JslAnnoId('trainable_medical_ner')
    NER_CHUNKER = JslAnnoId('ner_chunker')
    NER_CONVERTER_INTERNAL = JslAnnoId('ner_converter_internal')
    NER_DISAMBIGUATOR = JslAnnoId('ner_disambiguator')
    TRAINABLE_NER_DISAMBIGUATOR = JslAnnoId('trainable_ner_disambiguator')  # TODO feature node !!!
    RELATION_NER_CHUNKS_FILTERER = JslAnnoId('relation_ner_chunks_filterer')
    RE_IDENTIFICATION = JslAnnoId('re_identification')
    RELATION_EXTRACTION = JslAnnoId('relation_extraction')
    TRAINABLE_RELATION_EXTRACTION = JslAnnoId('trainable_relation_extraction')
    RELATION_EXTRACTION_DL = JslAnnoId('relation_extraction_dl')
    # TRAINABLE_RELATION_EXTRACTION_DL = JslAnnoId('trainable_relation_extraction_dl')
    SENTENCE_ENTITY_RESOLVER = JslAnnoId('sentence_entity_resolver')
    TRAINABLE_SENTENCE_ENTITY_RESOLVER = JslAnnoId('trainable_sentence_entity_resolver')
    MEDICAL_BERT_FOR_TOKEN_CLASSIFICATION = JslAnnoId('medical_bert_for_token_classification')


class OCR_NODE_IDS:
    """All available Feature nodes in OCR
    Used to cast the pipeline dependency resolution algorithm into an abstract graph
    """
    # Visual Document Understanding
    VISUAL_DOCUMENT_CLASSIFIER = JslAnnoId('visual_document_classifier')
    VISUAL_DOCUMENT_NER = JslAnnoId('visual_document_NER')

    # Object Detection
    IMAGE_HANDWRITTEN_DETECTOR = JslAnnoId('image_handwritten_detector')

    # TABLE Processors/Recognition
    IMAGE_TABLE_DETECTOR = JslAnnoId('image_table_detector')
    IMAGE_TABLE_CELL_DETECTOR = JslAnnoId('image_table_cell_detector')
    IMAGE_TABLE_CELL2TEXT_TABLE = JslAnnoId('image_table_cell2text_table')

    # PDF Processing
    PDF2TEXT = JslAnnoId('pdf2text')
    PDF2IMAGE = JslAnnoId('pdf2image')
    IMAGE2PDF = JslAnnoId('image2pdf')
    TEXT2PDF = JslAnnoId('text2pdf')
    PDF_ASSEMBLER = JslAnnoId('pdf_assembler')
    PDF_DRAW_REGIONS = JslAnnoId('pdf_draw_regions')
    PDF2TEXT_TABLE = JslAnnoId('pdf2text_table')

    # DOCX Processing
    DOC2TEXT = JslAnnoId('doc2text')
    DOC2TEXT_TABLE = JslAnnoId('doc2text_table')
    DOC2PDF = JslAnnoId('doc2pdf')
    PPT2TEXT_TABLE = JslAnnoId('ppt2text_table')
    PPT2PDF = JslAnnoId('ppt2pdf')

    # DICOM Processing
    DICOM2IMAGE = JslAnnoId('dicom2image')
    IMAGE2DICOM = JslAnnoId('IMAGE2DICOM')

    # Image Pre-Processing
    BINARY2IMAGE = JslAnnoId('binary2image')
    GPU_IMAGE_TRANSFORMER = JslAnnoId('GPU_IMAGE_TRANSFORMER')
    IMAGE_BINARIZER = JslAnnoId('image_binarizer')
    IMAGE_ADAPTIVE_BINARIZER = JslAnnoId('image_adaptive_binarizer')
    IMAGE_ADAPTIVE_THRESHOLDING = JslAnnoId('image_adaptive_thresholding')
    IMAGE_SCALER = JslAnnoId('image_scaler')
    IMAGE_ADAPTIVE_SCALER = JslAnnoId('image_adaptive_scaler')
    IMAGE_SKEW_CORRECTOR = JslAnnoId('image_skew_corrector')
    IMAGE_NOISE_SCORER = JslAnnoId('image_noise_scorer')
    IMAGE_REMOVE_OBJECTS = JslAnnoId('image_remove_objects')
    IMAGE_MORPHOLOGY_OPERATION = JslAnnoId('image_morphology_operation')
    IMAGE_CROPPER = JslAnnoId('image_cropper')
    IMAGE2REGION = JslAnnoId('image2region')
    IMAGE_LAYOUT_ANALZYER = JslAnnoId('image_layout_analyzer')
    IMAGE_SPLIT_REGIONS = JslAnnoId('image_split_regions')
    IMAGE_DRAW_REGIONS = JslAnnoId('image_draw_regions')

    # Character Recognition
    IMAGE2TEXT = JslAnnoId('image2text')
    IMAGE2TEXTPDF = JslAnnoId('image2textpdf')
    IMAGE2HOCR = JslAnnoId('image2hocr')
    IMAGE_BRANDS2TEXT = JslAnnoId('image_brands2text')

    # Other
    POSITION_FINDER = JslAnnoId('position_finder')
    UPDATE_TEXT_POSITION = JslAnnoId('update_text_position')
    FOUNDATION_ONE_REPORT_PARSER = JslAnnoId('foundation_one_report_parser')
    HOCR_DOCUMENT_ASSEMBLER = JslAnnoId('hocr_document_assembler')
    HOCR_TOKENIZER = JslAnnoId('hocr_tokenizer')

