from typing import Dict

from nlu.universe.atoms import JslAnnoId, JslAnnoPyClass
from nlu.universe.feature_node_ids import OCR_NODE_IDS, NLP_NODE_IDS, NLP_HC_NODE_IDS


class AnnoClassRef:
    # Reference of every Annotator class name in OS/HC/OCR
    # Maps JslAnnoID to ClassNames in Python/Java from Spark NLP/Healthcare/ OCR
    A_O = OCR_NODE_IDS
    A_H = None  # NLP_HC_ANNO
    A_N = NLP_NODE_IDS
    HC_A_N = NLP_HC_NODE_IDS
    # Map AnnoID to PyCLass
    JSL_anno2_py_class: Dict[JslAnnoId, JslAnnoPyClass] = {
        A_N.COREF_SPAN_BERT: 'SpanBertCorefModel',
        A_N.PARTIALLY_IMPLEMENTED: 'PartiallyIntegrated',

        A_N.BIG_TEXT_MATCHER: 'BigTextMatcher',
        A_N.CHUNK2DOC: 'Chunk2Doc',
        A_N.CHUNK_EMBEDDINGS_CONVERTER: 'ChunkEmbeddings',
        A_N.CHUNK_TOKENIZER: 'Tokenizer',
        A_N.CHUNKER: 'Chunker',
        A_N.CLASSIFIER_DL: 'ClassifierDLModel',
        A_N.CONTEXT_SPELL_CHECKER: 'ContextSpellCheckerModel',
        A_N.DATE_MATCHER: 'DateMatcher',
        A_N.UNTYPED_DEPENDENCY_PARSER: 'DependencyParserModel',
        A_N.TYPED_DEPENDENCY_PARSER: 'TypedDependencyParserModel',
        A_N.DOC2CHUNK: 'Doc2Chunk',
        A_N.DOC2VEC: 'Doc2VecModel',
        A_N.TRAIANBLE_DOC2VEC: 'Doc2VecApproach',

        A_N.MULTI_DOCUMENT_ASSEMBLER : 'MultiDocumentAssembler',
        A_N.ALBERT_FOR_QUESTION_ANSWERING : 'AlbertForQuestionAnswering',
        A_N.BERT_FOR_QUESTION_ANSWERING : 'BertForQuestionAnswering',
        A_N.DE_BERTA_FOR_QUESTION_ANSWERING : 'DeBertaForQuestionAnswering',
        A_N.DISTIL_BERT_FOR_QUESTION_ANSWERING : 'DistilBertForQuestionAnswering',
        A_N.LONGFORMER_FOR_QUESTION_ANSWERING : 'LongformerForQuestionAnswering',
        A_N.ROBERTA_FOR_QUESTION_ANSWERING : 'RoBertaForQuestionAnswering',
        A_N.XLM_ROBERTA_FOR_QUESTION_ANSWERING : 'XlmRoBertaForQuestionAnswering',




        A_N.DOCUMENT_ASSEMBLER: 'DocumentAssembler',
        A_N.DOCUMENT_NORMALIZER: 'DocumentNormalizer',
        A_N.EMBEDDINGS_FINISHER: 'EmbeddingsFinisher',
        A_N.ENTITY_RULER: 'EntityRulerModel',
        A_N.FINISHER: 'Finisher',
        A_N.GRAPH_EXTRACTION: 'GraphExtraction',
        A_N.GRAPH_FINISHER: 'GraphFinisher',
        A_N.LANGUAGE_DETECTOR_DL: 'LanguageDetectorDL',
        A_N.LEMMATIZER: 'LemmatizerModel',
        A_N.MULTI_CLASSIFIER_DL: 'MultiClassifierDLModel',
        A_N.MULTI_DATE_MATCHER: 'MultiDateMatcher',
        A_N.N_GRAMM_GENERATOR: 'NGramGenerator',
        A_N.NER_CONVERTER: 'NerConverter',
        A_N.NER_CRF: 'NerCrfModel',
        A_N.NER_DL: 'NerDLModel',
        A_N.NER_OVERWRITER: 'NerOverwriter',
        A_N.NORMALIZER: 'NormalizerModel',
        A_N.NORVIG_SPELL_CHECKER: 'NorvigSweetingModel',
        A_N.POS: 'PerceptronModel',
        A_N.RECURISVE_TOKENIZER: 'RecursiveTokenizerModel',

        A_N.REGEX_MATCHER: 'RegexMatcherModel',
        A_N.REGEX_TOKENIZER: 'RegexTokenizer',
        A_N.SENTENCE_DETECTOR: 'SentenceDetector',
        A_N.SENTENCE_DETECTOR_DL: 'SentenceDetectorDLModel',
        A_N.SENTENCE_EMBEDDINGS_CONVERTER: 'SentenceEmbeddings',
        A_N.STEMMER: 'Stemmer',
        A_N.STOP_WORDS_CLEANER: 'StopWordsCleaner',
        A_N.SYMMETRIC_DELETE_SPELLCHECKER: 'SymmetricDeleteModel',
        A_N.TEXT_MATCHER: 'TextMatcherModel',
        A_N.TOKEN2CHUNK: 'Token2Chunk',
        A_N.TOKEN_ASSEMBLER: 'TokenAssembler',
        A_N.TOKENIZER: 'TokenizerModel',
        A_N.SENTIMENT_DL: 'SentimentDLModel',
        A_N.SENTIMENT_DETECTOR: 'SentimentDetectorModel',
        A_N.VIVEKN_SENTIMENT: 'ViveknSentimentModel',
        A_N.WORD_EMBEDDINGS: 'WordEmbeddingsModel',
        A_N.WORD_SEGMENTER: 'WordSegmenterModel',
        A_N.YAKE_KEYWORD_EXTRACTION: 'YakeKeywordExtraction',
        A_N.ALBERT_EMBEDDINGS: 'AlbertEmbeddings',
        A_N.ALBERT_FOR_TOKEN_CLASSIFICATION: 'AlbertForTokenClassification',
        A_N.BERT_EMBEDDINGS: 'BertEmbeddings',
        A_N.BERT_FOR_TOKEN_CLASSIFICATION: 'BertForTokenClassification',
        A_N.BERT_SENTENCE_EMBEDDINGS: 'BertSentenceEmbeddings',
        A_N.DISTIL_BERT_EMBEDDINGS: 'DistilBertEmbeddings',
        A_N.DISTIL_BERT_FOR_SEQUENCE_CLASSIFICATION: 'DistilBertForSequenceClassification',
        A_N.BERT_FOR_SEQUENCE_CLASSIFICATION: 'BertForSequenceClassification',
        A_N.ELMO_EMBEDDINGS: 'ElmoEmbeddings',
        A_N.LONGFORMER_EMBEDDINGS: 'LongformerEmbeddings',
        A_N.LONGFORMER_FOR_TOKEN_CLASSIFICATION: 'LongformerForTokenClassification',
        A_N.MARIAN_TRANSFORMER: 'MarianTransformer',
        A_N.ROBERTA_EMBEDDINGS: 'RoBertaEmbeddings',
        A_N.ROBERTA_FOR_TOKEN_CLASSIFICATION: 'RoBertaForTokenClassification',
        A_N.ROBERTA_SENTENCE_EMBEDDINGS: 'RoBertaSentenceEmbeddings',
        A_N.T5_TRANSFORMER: 'T5Transformer',
        A_N.UNIVERSAL_SENTENCE_ENCODER: 'UniversalSentenceEncoder',
        A_N.XLM_ROBERTA_EMBEDDINGS: 'XlmRoBertaEmbeddings',
        A_N.XLM_ROBERTA_FOR_TOKEN_CLASSIFICATION: 'XlmRoBertaForTokenClassification',
        A_N.XLM_ROBERTA_SENTENCE_EMBEDDINGS: 'XlmRoBertaSentenceEmbeddings',
        A_N.XLNET_EMBEDDINGS: 'XlnetEmbeddings',
        A_N.XLNET_FOR_TOKEN_CLASSIFICATION: 'XlnetForTokenClassification',
        A_N.XLM_ROBERTA_FOR_SEQUENCE_CLASSIFICATION: 'XlmRoBertaForSequenceClassification',
        A_N.ROBERTA_FOR_SEQUENCE_CLASSIFICATION: 'RoBertaForSequenceClassification',
        A_N.LONGFORMER_FOR_SEQUENCE_CLASSIFICATION: 'LongformerForSequenceClassification',
        A_N.ALBERT_FOR_SEQUENCE_CLASSIFICATION: 'AlbertForSequenceClassification',
        A_N.XLNET_FOR_SEQUENCE_CLASSIFICATION: 'XlnetForSequenceClassification',
        A_N.GPT2: 'GPT2Transformer',
        A_N.DEBERTA_WORD_EMBEDDINGS: 'DeBertaEmbeddings',
        A_N.DEBERTA_FOR_TOKEN_CLASSIFICATION : 'DeBertaForTokenClassification',
        A_N.CAMENBERT_EMBEDDINGS : 'CamemBertEmbeddings',

        A_N.TRAINABLE_VIVEKN_SENTIMENT: 'ViveknSentimentApproach',
        A_N.TRAINABLE_SENTIMENT: 'SentimentDetector',
        A_N.TRAINABLE_SENTIMENT_DL: 'SentimentDLApproach',
        A_N.TRAINABLE_CLASSIFIER_DL: 'ClassifierDLApproach',
        A_N.TRAINABLE_MULTI_CLASSIFIER_DL: 'MultiClassifierDLApproach',
        A_N.TRAINABLE_NER_DL: 'NerDLApproach',
        A_N.TRAINABLE_NER_CRF: 'NerCrfApproach',

        A_N.TRAINABLE_POS: 'PerceptronApproach',
        A_N.TRAINABLE_DEP_PARSE_TYPED: 'TypedDependencyParserApproach',
        A_N.TRAINABLE_DEP_PARSE_UN_TYPED: 'DependencyParserApproach',
        A_N.TRAINABLE_DOC2VEC: 'Doc2VecApproach',
        A_N.TRAINABLE_ENTITY_RULER: 'EntityRulerApproach',
        A_N.TRAINABLE_LEMMATIZER: 'Lemmatizer',
        A_N.TRAINABLE_NORMALIZER: 'Normalizer',
        A_N.TRAINABLE_NORVIG_SPELL_CHECKER: 'NorvigSweetingApproach',
        A_N.TRAINABLE_RECURISVE_TOKENIZER: 'RecursiveTokenizer',
        A_N.TRAINABLE_REGEX_MATCHER: 'RegexMatcher',
        A_N.TRAINABLE_SENTENCE_DETECTOR_DL: 'SentenceDetectorDLApproach',
        A_N.TRAINABLE_WORD_EMBEDDINGS: 'WordEmbeddings',
        A_N.TRAINABLE_SYMMETRIC_DELETE_SPELLCHECKER: 'SymmetricDeleteApproach',
        A_N.TRAINABLE_TEXT_MATCHER: 'TextMatcher',
        A_N.TRAINABLE_TOKENIZER: 'Tokenizer',
        A_N.TRAINABLE_WORD_SEGMENTER: 'WordSegmenterApproach',

        A_N.DISTIL_BERT_FOR_TOKEN_CLASSIFICATION: 'DistilBertForTokenClassification',
        A_N.WORD_2_VEC: 'Word2VecModel',
        A_N.DEBERTA_FOR_SEQUENCE_CLASSIFICATION: 'DeBertaForSequenceClassification',

        A_N.BERT_SENTENCE_CHUNK_EMBEDDINGS : 'BertSentenceChunkEmbeddings',


        A_N.PARTIAL_AssertionFilterer : 'AssertionFilterer',
        A_N.PARTIAL_ChunkConverter : 'ChunkConverter',
        A_N.PARTIAL_ChunkKeyPhraseExtraction : 'ChunkKeyPhraseExtraction',
        A_N.PARTIAL_ChunkSentenceSplitter : 'ChunkSentenceSplitter',
        A_N.PARTIAL_ChunkFiltererApproach : 'ChunkFiltererApproach',
        A_N.PARTIAL_ChunkFiltererApproach : 'ChunkFiltererApproach',
        A_N.PARTIAL_ChunkFilterer : 'ChunkFilterer',
        A_N.PARTIAL_ChunkMapperApproach : 'ChunkMapperApproach',
        A_N.PARTIAL_ChunkMapperApproach : 'ChunkMapperApproach',
        A_N.PARTIAL_ChunkMapperFilterer : 'ChunkMapperFilterer',
        A_N.PARTIAL_DocumentLogRegClassifierApproach : 'DocumentLogRegClassifierApproach',
        A_N.PARTIAL_DocumentLogRegClassifierApproach : 'DocumentLogRegClassifierApproach',
        A_N.PARTIAL_DocumentLogRegClassifierModel : 'DocumentLogRegClassifierModel',
        A_N.PARTIAL_ContextualParserApproach : 'ContextualParserApproach',
        A_N.PARTIAL_ContextualParserApproach : 'ContextualParserApproach',
        A_N.PARTIAL_ReIdentification : 'ReIdentification',
        A_N.PARTIAL_NerDisambiguator : 'NerDisambiguator',
        A_N.PARTIAL_NerDisambiguatorModel : 'NerDisambiguatorModel',
        A_N.PARTIAL_AverageEmbeddings : 'AverageEmbeddings',
        A_N.PARTIAL_EntityChunkEmbeddings : 'EntityChunkEmbeddings',
        A_N.PARTIAL_ChunkMergeApproach : 'ChunkMergeApproach',
        A_N.PARTIAL_ChunkMergeApproach : 'ChunkMergeApproach',
        A_N.PARTIAL_IOBTagger : 'IOBTagger',
        A_N.PARTIAL_NerChunker : 'NerChunker',
        A_N.PARTIAL_NerConverterInternalModel : 'NerConverterInternalModel',
        A_N.PARTIAL_DateNormalizer : 'DateNormalizer',
        A_N.PARTIAL_PosologyREModel : 'PosologyREModel',
        A_N.PARTIAL_RENerChunksFilter : 'RENerChunksFilter',
        A_N.PARTIAL_ResolverMerger : 'ResolverMerger',
        A_N.PARTIAL_AnnotationMerger : 'AnnotationMerger',
        A_N.PARTIAL_Router : 'Router',
        A_N.PARTIAL_Word2VecApproach : 'Word2VecApproach',
        A_N.PARTIAL_Word2VecApproach : 'Word2VecApproach',
        A_N.PARTIAL_WordEmbeddings : 'WordEmbeddings',
        A_N.PARTIAL_EntityRulerApproach : 'EntityRulerApproach',
        A_N.PARTIAL_EntityRulerApproach : 'EntityRulerApproach',
        A_N.PARTIAL_EntityRulerModel : 'EntityRulerModel',
        A_N.PARTIAL_TextMatcherModel : 'TextMatcherModel',
        A_N.PARTIAL_BigTextMatcher : 'BigTextMatcher',
        A_N.PARTIAL_BigTextMatcherModel : 'BigTextMatcherModel',
        A_N.PARTIAL_DateMatcher : 'DateMatcher',
        A_N.PARTIAL_MultiDateMatcher : 'MultiDateMatcher',
        A_N.PARTIAL_RegexMatcher : 'RegexMatcher',
        A_N.PARTIAL_TextMatcher : 'TextMatcher',
        A_N.PARTIAL_NerApproach : 'NerApproach',
        A_N.PARTIAL_NerCrfApproach : 'NerCrfApproach',
        A_N.PARTIAL_NerCrfApproach : 'NerCrfApproach',
        A_N.PARTIAL_NerCrfApproach : 'NerCrfApproach',
        A_N.PARTIAL_NerOverwriter : 'NerOverwriter',
        A_N.PARTIAL_DependencyParserApproach : 'DependencyParserApproach',
        A_N.PARTIAL_DependencyParserApproach : 'DependencyParserApproach',
        A_N.PARTIAL_TypedDependencyParserApproach : 'TypedDependencyParserApproach',
        A_N.PARTIAL_TypedDependencyParserApproach : 'TypedDependencyParserApproach',
        A_N.PARTIAL_SentenceDetectorDLApproach : 'SentenceDetectorDLApproach',
        A_N.PARTIAL_SentenceDetectorDLApproach : 'SentenceDetectorDLApproach',
        A_N.PARTIAL_SentimentDetector : 'SentimentDetector',
        A_N.PARTIAL_ViveknSentimentApproach : 'ViveknSentimentApproach',
        A_N.PARTIAL_ViveknSentimentApproach : 'ViveknSentimentApproach',
        A_N.PARTIAL_ContextSpellCheckerApproach : 'ContextSpellCheckerApproach',
        A_N.PARTIAL_ContextSpellCheckerApproach : 'ContextSpellCheckerApproach',
        A_N.PARTIAL_NorvigSweetingApproach : 'NorvigSweetingApproach',
        A_N.PARTIAL_NorvigSweetingApproach : 'NorvigSweetingApproach',
        A_N.PARTIAL_SymmetricDeleteApproach : 'SymmetricDeleteApproach',
        A_N.PARTIAL_SymmetricDeleteApproach : 'SymmetricDeleteApproach',
        A_N.PARTIAL_ChunkTokenizer : 'ChunkTokenizer',
        A_N.PARTIAL_ChunkTokenizerModel : 'ChunkTokenizerModel',
        A_N.PARTIAL_RecursiveTokenizer : 'RecursiveTokenizer',
        A_N.PARTIAL_RecursiveTokenizerModel : 'RecursiveTokenizerModel',
        A_N.PARTIAL_Token2Chunk : 'Token2Chunk',
        A_N.PARTIAL_WordSegmenterApproach : 'WordSegmenterApproach',
        A_N.PARTIAL_WordSegmenterApproach : 'WordSegmenterApproach',
        A_N.PARTIAL_GraphExtraction : 'GraphExtraction',
        A_N.PARTIAL_Lemmatizer : 'Lemmatizer',
        A_N.PARTIAL_Normalizer : 'Normalizer',



































    }
    JSL_anno_HC_ref_2_py_class: Dict[JslAnnoId, JslAnnoPyClass] = {

        HC_A_N.CHUNK_MAPPER_MODEL: 'ChunkMapperModel',
        HC_A_N.ASSERTION_DL: 'AssertionDLModel',
        HC_A_N.TRAINABLE_ASSERTION_DL: 'AssertionDLApproach',
        HC_A_N.ASSERTION_FILTERER: 'AssertionFilterer',
        HC_A_N.ASSERTION_LOG_REG: 'AssertionLogRegModel',
        HC_A_N.TRAINABLE_ASSERTION_LOG_REG: 'AssertionLogRegApproach',
        HC_A_N.CHUNK2TOKEN: 'Chunk2Token',
        HC_A_N.CHUNK_ENTITY_RESOLVER: '',  # DEPRECATED
        HC_A_N.TRAINABLE_CHUNK_ENTITY_RESOLVER: '',  # DEPRECATED
        HC_A_N.CHUNK_FILTERER: 'ChunkFilterer',
        HC_A_N.TRAINABLE_CHUNK_FILTERER: 'ChunkFiltererApproach',
        HC_A_N.CHUNK_KEY_PHRASE_EXTRACTION: 'ChunkKeyPhraseExtraction',
        HC_A_N.CHUNK_MERGE: 'ChunkMergeModel',
        HC_A_N.TRAINABLE_CHUNK_MERGE: 'ChunkMergeApproach',
        HC_A_N.CONTEXTUAL_PARSER: 'ContextualParserModel',
        HC_A_N.TRAIANBLE_CONTEXTUAL_PARSER: 'ContextualParserApproach',
        HC_A_N.DE_IDENTIFICATION: 'DeIdentificationModel',
        HC_A_N.TRAINABLE_DE_IDENTIFICATION: 'DeIdentification',
        HC_A_N.DOCUMENT_LOG_REG_CLASSIFIER: 'DocumentLogRegClassifierModel',
        HC_A_N.TRAINABLE_DOCUMENT_LOG_REG_CLASSIFIER: 'DocumentLogRegClassifierApproach',
        HC_A_N.DRUG_NORMALIZER: 'DrugNormalizer',
        # HC_A_N.FEATURES_ASSEMBLER  : '', # TODO spark calss>?
        HC_A_N.GENERIC_CLASSIFIER: 'GenericClassifierModel',
        HC_A_N.TRAINABLE_GENERIC_CLASSIFIER: 'GenericClassifierApproach',
        HC_A_N.IOB_TAGGER: 'IOBTagger',
        HC_A_N.MEDICAL_NER: 'MedicalNerModel',
        HC_A_N.TRAINABLE_MEDICAL_NER: 'MedicalNerApproach',
        HC_A_N.NER_CHUNKER: 'NerChunker',
        HC_A_N.NER_CONVERTER_INTERNAL: 'NerConverterInternal',
        HC_A_N.NER_DISAMBIGUATOR: 'NerDisambiguatorModel',
        HC_A_N.TRAINABLE_NER_DISAMBIGUATOR: 'NerDisambiguatorModel',
        HC_A_N.RELATION_NER_CHUNKS_FILTERER: 'NerDisambiguator',
        HC_A_N.RE_IDENTIFICATION: 'ReIdentification',
        HC_A_N.RELATION_EXTRACTION: 'RelationExtractionModel',
        HC_A_N.TRAINABLE_RELATION_EXTRACTION: 'RelationExtractionApproach',
        HC_A_N.RELATION_EXTRACTION_DL: 'RelationExtractionDLModel',
        # HC_A_N.TRAINABLE_RELATION_EXTRACTION_DL  : '',
        HC_A_N.SENTENCE_ENTITY_RESOLVER: 'SentenceEntityResolverModel',
        HC_A_N.TRAINABLE_SENTENCE_ENTITY_RESOLVER: 'SentenceEntityResolverApproach',
        HC_A_N.MEDICAL_BERT_FOR_TOKEN_CLASSIFICATION: 'MedicalBertForTokenClassifier',

        HC_A_N.MEDICAL_BERT_FOR_SEQUENCE_CLASSIFICATION: 'MedicalBertForSequenceClassification',
        HC_A_N.MEDICAL_DISTILBERT_FOR_SEQUENCE_CLASSIFICATION: 'MedicalDistilBertForSequenceClassification',
        HC_A_N.ENTITY_CHUNK_EMBEDDING: 'EntityChunkEmbeddings',
        HC_A_N.ZERO_SHOT_RELATION_EXTRACTION: 'ZeroShotRelationExtractionModel',

    }
    JSL_anno_OCR_ref_2_py_class: Dict[JslAnnoId, JslAnnoPyClass] = {
        OCR_NODE_IDS.IMAGE2TEXT: 'ImageToText',
        OCR_NODE_IDS.PDF2TEXT: 'PdfToText',
        OCR_NODE_IDS.DOC2TEXT: 'DocToText',
        OCR_NODE_IDS.BINARY2IMAGE: 'BinaryToImage',
        OCR_NODE_IDS.PDF2TEXT_TABLE: 'PdfToTextTable',
        OCR_NODE_IDS.PPT2TEXT_TABLE: 'PptToTextTable',
        OCR_NODE_IDS.DOC2TEXT_TABLE: 'DocToTextTable',
        OCR_NODE_IDS.TEXT2PDF: 'TextToPdf',
        OCR_NODE_IDS.VISUAL_DOCUMENT_CLASSIFIER: 'VisualDocumentClassifier',
        OCR_NODE_IDS.IMAGE2HOCR: 'ImageToHocr',

    }



    @staticmethod
    def get_os_pyclass_2_anno_id_dict():
        # Flipped, maps PyClass to AnnoID
        JSL_py_class_2_anno_id: Dict[JslAnnoPyClass, JslAnnoId] = {AnnoClassRef.JSL_anno2_py_class[k]: k for k in
                                                                   AnnoClassRef.JSL_anno2_py_class}
        return JSL_py_class_2_anno_id

    @staticmethod
    def get_hc_pyclass_2_anno_id_dict():
        # Flipped, maps PyClass to AnnoID
        JSL_HC_py_class_2_anno_id: Dict[JslAnnoId, JslAnnoPyClass] = {AnnoClassRef.JSL_anno_HC_ref_2_py_class[k]: k for
                                                                      k in AnnoClassRef.JSL_anno_HC_ref_2_py_class}
        return JSL_HC_py_class_2_anno_id

    @staticmethod
    def get_ocr_pyclass_2_anno_id_dict():
        # Flipped, maps PyClass to AnnoID
        JSL_OCR_py_class_2_anno_id: Dict[JslAnnoId, JslAnnoPyClass] = {AnnoClassRef.JSL_anno_OCR_ref_2_py_class[k]: k
                                                                       for
                                                                       k in AnnoClassRef.JSL_anno_OCR_ref_2_py_class}
        return JSL_OCR_py_class_2_anno_id


# Flipped, maps PyClass to AnnoID
AnnoClassRef.JSL_OS_py_class_2_anno_id: Dict[JslAnnoPyClass, JslAnnoId] = {AnnoClassRef.JSL_anno2_py_class[k]: k for k in AnnoClassRef.JSL_anno2_py_class}
AnnoClassRef.JSL_HC_py_class_2_anno_id: Dict[JslAnnoId, JslAnnoPyClass] = {AnnoClassRef.JSL_anno_HC_ref_2_py_class[k]: k for k in
                                                                           AnnoClassRef.JSL_anno_HC_ref_2_py_class}
AnnoClassRef.JSL_OCR_py_class_2_anno_id: Dict[JslAnnoId, JslAnnoPyClass] = {AnnoClassRef.JSL_anno_OCR_ref_2_py_class[k]: k for k in
                                                                            AnnoClassRef.JSL_anno_OCR_ref_2_py_class}