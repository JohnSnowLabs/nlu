"""
Collection of universes shared across all libraries (NLP/HC/OCR), which are collections of atoms
"""
from dataclasses import dataclass
from typing import List, Union

from nlu.universe.atoms import JslAnnoId, JslFeature, ExternalFeature
from nlu.universe.feature_node_ids import NLP_NODE_IDS, OCR_NODE_IDS, NLP_HC_NODE_IDS
from nlu.universe.feature_universes import NLP_FEATURES, OCR_FEATURES, NLP_HC_FEATURES


### ____ Pipeline Graph Representation Logik Building Blocks ______

@dataclass
class FeatureNode:  # or Mode Node?
    """Defines a Node in a ML Dependency Feature Graph.
    Anno= Node, In = In arrows, Out = out Arrows
    Each NLU Component will output one of these
    NODE = Anno Class
    INPUTS = Array of ML-Features
    OUTPUTS = Array of ML-Features

    Used to cast the pipeline dependency resolution algorithm into an abstract grpah
    """

    JSL_generator_anno_class: JslAnnoId  # JSL Annotator that can generate this Triplet. Could be from OCR/JSL-Internal/Spark-NLP
    ins: List[JslFeature]  # JSL Annotator that can generate this Triplet
    outs: List[JslFeature]  # JSL Annotator that can generate this Triplet


@dataclass
class NlpFeatureNode:  # or Mode Node? (FeatureNode)
    """A node representation for a Spark OCR Annotator
    Used to cast the pipeline dependency resolution algorithm into an abstract grpah
    """
    node: Union[JslAnnoId]  # JSL Annotator that can generate this Triplet. Could be from OCR/JSL-Internal/Spark-NLP
    ins: List[JslFeature]  # JSL Annotator that can generate this Triplet
    outs: List[JslFeature]  # JSL Annotator that can generate this Triplet


class NlpHcFeatureNode(FeatureNode): pass


class OcrFeatureNode(FeatureNode): pass


class EXTERNAL_NODES():
    """
    Start Node definitions for the NLU Pipeline Graph completion logic
    These are analogus to the various input types NLU may accept
    """
    RAW_TEXT = ExternalFeature('text')
    NON_WHITESPACED_TEXT = ExternalFeature('non_whitespaced_text')  # i.e. Chinese, Russian, etc..

    # TODO define how its derivable, i.e Accepted input types that can be converted to spark DF types
    # str_array = 'str_array'
    #
    # pandas_df = 'pandas_df'
    # pd_series = 'pandas_series'
    #
    # np_array = 'pandas_series'
    #
    # img_path = 'pandas_series'
    # file_path = 'file_path' # todo more granuar, i.e. by file type?


@dataclass
class NLP_FEATURE_NODES:  # or Mode Node?
    """All avaiable Feature nodes in Spark NLP
    Used to cast the pipeline dependency resolution algorithm into an abstract grpah
    """
    # High Level NLP Feature Nodes
    E = EXTERNAL_NODES
    A = NLP_NODE_IDS
    F = NLP_FEATURES
    nodes = {
        A.PARTIALLY_IMPLEMENTED: NlpFeatureNode(A.PARTIALLY_IMPLEMENTED, [F.UNKOWN], [F.UNKOWN]),

        A.COREF_SPAN_BERT: NlpFeatureNode(A.COREF_SPAN_BERT, [F.DOCUMENT, F.TOKEN], [F.COREF_TOKEN]),


        A.BIG_TEXT_MATCHER: NlpFeatureNode(A.BIG_TEXT_MATCHER, [F.DOCUMENT, F.TOKEN], [F.CHUNK]),
        A.CHUNK2DOC: NlpFeatureNode(A.CHUNK2DOC, [F.NAMED_ENTITY_CONVERTED], [F.DOCUMENT_FROM_CHUNK]),
        A.CHUNK_EMBEDDINGS_CONVERTER: NlpFeatureNode(A.CHUNK_EMBEDDINGS_CONVERTER, [F.CHUNK, F.WORD_EMBEDDINGS],
                                                     [F.CHUNK_EMBEDDINGS]),
        A.CHUNK_TOKENIZER: NlpFeatureNode(A.CHUNK_TOKENIZER, [F.CHUNK], [F.TOKEN_CHUNKED]),
        A.CHUNKER: NlpFeatureNode(A.CHUNKER, [F.DOCUMENT, F.POS], [F.CHUNK]),
        A.CLASSIFIER_DL: NlpFeatureNode(A.CLASSIFIER_DL, [F.SENTENCE_EMBEDDINGS], [F.CATEGORY]),
        A.TRAINABLE_CLASSIFIER_DL: NlpFeatureNode(A.CLASSIFIER_DL, [F.SENTENCE_EMBEDDINGS], [F.CATEGORY]),
        A.CONTEXT_SPELL_CHECKER: NlpFeatureNode(A.CONTEXT_SPELL_CHECKER, [F.TOKEN], [F.TOKEN_SPELL_CHECKED]),
        A.DATE_MATCHER: NlpFeatureNode(A.DATE_MATCHER, [F.DOCUMENT], [F.DATE]),
        A.UNTYPED_DEPENDENCY_PARSER: NlpFeatureNode(A.UNTYPED_DEPENDENCY_PARSER, [F.DOCUMENT, F.POS, F.TOKEN],
                                                    [F.UNLABLED_DEPENDENCY]),
        A.TYPED_DEPENDENCY_PARSER: NlpFeatureNode(A.TYPED_DEPENDENCY_PARSER, [F.TOKEN, F.POS, F.UNLABLED_DEPENDENCY],
                                                  [F.LABELED_DEPENDENCY]),
        A.DOC2CHUNK: NlpFeatureNode(A.DOC2CHUNK, [F.DOCUMENT], [F.DOCUMENT_FROM_CHUNK]),


        A.MULTI_DOCUMENT_ASSEMBLER: NlpFeatureNode(A.MULTI_DOCUMENT_ASSEMBLER, [F.RAW_QUESTION, F.RAW_QUESTION_CONTEXT], [F.DOCUMENT_QUESTION, F.DOCUMENT_QUESTION_CONTEXT]),
        A.ALBERT_FOR_QUESTION_ANSWERING: NlpFeatureNode(A.ALBERT_FOR_QUESTION_ANSWERING, [F.DOCUMENT_QUESTION, F.DOCUMENT_QUESTION_CONTEXT], [F.CLASSIFIED_SPAN]),
        A.BERT_FOR_QUESTION_ANSWERING: NlpFeatureNode(A.BERT_FOR_QUESTION_ANSWERING, [F.DOCUMENT_QUESTION, F.DOCUMENT_QUESTION_CONTEXT], [F.CLASSIFIED_SPAN]),
        A.DE_BERTA_FOR_QUESTION_ANSWERING: NlpFeatureNode(A.DE_BERTA_FOR_QUESTION_ANSWERING, [F.DOCUMENT_QUESTION, F.DOCUMENT_QUESTION_CONTEXT], [F.CLASSIFIED_SPAN]),
        A.DISTIL_BERT_FOR_QUESTION_ANSWERING: NlpFeatureNode(A.DISTIL_BERT_FOR_QUESTION_ANSWERING, [F.DOCUMENT_QUESTION, F.DOCUMENT_QUESTION_CONTEXT], [F.CLASSIFIED_SPAN]),
        A.LONGFORMER_FOR_QUESTION_ANSWERING: NlpFeatureNode(A.LONGFORMER_FOR_QUESTION_ANSWERING, [F.DOCUMENT_QUESTION, F.DOCUMENT_QUESTION_CONTEXT], [F.CLASSIFIED_SPAN]),
        A.ROBERTA_FOR_QUESTION_ANSWERING: NlpFeatureNode(A.ROBERTA_FOR_QUESTION_ANSWERING, [F.DOCUMENT_QUESTION, F.DOCUMENT_QUESTION_CONTEXT], [F.CLASSIFIED_SPAN]),
        A.XLM_ROBERTA_FOR_QUESTION_ANSWERING: NlpFeatureNode(A.XLM_ROBERTA_FOR_QUESTION_ANSWERING, [F.DOCUMENT_QUESTION, F.DOCUMENT_QUESTION_CONTEXT], [F.CLASSIFIED_SPAN]),




        A.DOCUMENT_ASSEMBLER: NlpFeatureNode(A.DOCUMENT_ASSEMBLER, [E.RAW_TEXT], [F.DOCUMENT]),
        A.DOCUMENT_NORMALIZER: NlpFeatureNode(A.DOCUMENT_NORMALIZER, [F.DOCUMENT], [F.DOCUMENT_GENERATED]),
        A.EMBEDDINGS_FINISHER: NlpFeatureNode(A.EMBEDDINGS_FINISHER, [F.ANY_EMBEDDINGS], [F.FINISHED_EMBEDDINGS]),
        # A.# ENTITY_RULER : NlpFeatureNode(A.ENTITY_RULER, [F.], [F.]) # TODO? ,
        A.FINISHER: NlpFeatureNode(A.FINISHER, [F.ANY], [F.ANY_FINISHED]),
        A.GRAPH_EXTRACTION: NlpFeatureNode(A.GRAPH_EXTRACTION, [F.DOCUMENT, F.TOKEN, F.NAMED_ENTITY_IOB], [F.NODE]),
        # A.# GRAPH_FINISHER : NlpFeatureNode(A.GRAPH_FINISHER, [F.], [F.]) ,
        A.LANGUAGE_DETECTOR_DL: NlpFeatureNode(A.LANGUAGE_DETECTOR_DL, [F.DOCUMENT], [F.LANGUAGE]),
        A.LEMMATIZER: NlpFeatureNode(A.LEMMATIZER, [F.TOKEN], [F.TOKEN_LEMATIZED]),
        A.MULTI_CLASSIFIER_DL: NlpFeatureNode(A.MULTI_CLASSIFIER_DL, [F.SENTENCE_EMBEDDINGS],
                                              [F.MULTI_DOCUMENT_CLASSIFICATION]),
        A.TRAINABLE_MULTI_CLASSIFIER_DL: NlpFeatureNode(A.MULTI_CLASSIFIER_DL, [F.SENTENCE_EMBEDDINGS],
                                                        [F.MULTI_DOCUMENT_CLASSIFICATION]),
        A.MULTI_DATE_MATCHER: NlpFeatureNode(A.MULTI_DATE_MATCHER, [F.DOCUMENT], [F.DATE]),
        A.N_GRAMM_GENERATOR: NlpFeatureNode(A.N_GRAMM_GENERATOR, [F.TOKEN], [F.CHUNK]),
        A.NER_CONVERTER: NlpFeatureNode(A.NER_CONVERTER, [F.TOKEN, F.DOCUMENT, F.NAMED_ENTITY_IOB],
                                        [F.NAMED_ENTITY_CONVERTED]),
        A.NER_CRF: NlpFeatureNode(A.NER_CRF, [F.DOCUMENT, F.TOKEN, F.WORD_EMBEDDINGS], [F.NAMED_ENTITY_IOB]),
        A.NER_DL: NlpFeatureNode(A.NER_DL, [F.DOCUMENT, F.TOKEN, F.WORD_EMBEDDINGS], [F.NAMED_ENTITY_IOB]),
        A.TRAINABLE_NER_DL: NlpFeatureNode(A.TRAINABLE_NER_DL, [F.DOCUMENT, F.TOKEN, F.WORD_EMBEDDINGS],
                                           [F.NAMED_ENTITY_IOB]),
        A.NER_OVERWRITER: NlpFeatureNode(A.NER_OVERWRITER, [F.NAMED_ENTITY_IOB], [F.NAMED_ENTITY_IOB]),
        A.NORMALIZER: NlpFeatureNode(A.NORMALIZER, [F.TOKEN], [F.TOKEN_NORMALIZED]),
        A.NORVIG_SPELL_CHECKER: NlpFeatureNode(A.NORVIG_SPELL_CHECKER, [F.TOKEN], [F.TOKEN_SPELL_CHECKED]),
        A.POS: NlpFeatureNode(A.POS, [F.TOKEN, F.DOCUMENT], [F.POS]),
        A.TRAINABLE_POS: NlpFeatureNode(A.POS, [F.TOKEN, F.DOCUMENT], [F.POS]),
        A.RECURISVE_TOKENIZER: NlpFeatureNode(A.RECURISVE_TOKENIZER, [F.DOCUMENT], [F.TOKEN]),
        A.REGEX_MATCHER: NlpFeatureNode(A.REGEX_MATCHER, [F.DOCUMENT], [F.NAMED_ENTITY_CONVERTED]),
        A.REGEX_TOKENIZER: NlpFeatureNode(A.REGEX_TOKENIZER, [F.DOCUMENT], [F.TOKEN]),
        A.SENTENCE_DETECTOR: NlpFeatureNode(A.SENTENCE_DETECTOR, [F.DOCUMENT], [F.SENTENCE]),
        A.SENTENCE_DETECTOR_DL: NlpFeatureNode(A.SENTENCE_DETECTOR_DL, [F.DOCUMENT], [F.SENTENCE]),
        A.SENTENCE_EMBEDDINGS_CONVERTER: NlpFeatureNode(A.SENTENCE_EMBEDDINGS_CONVERTER,
                                                        [F.DOCUMENT, F.WORD_EMBEDDINGS], [F.SENTENCE_EMBEDDINGS]),
        A.SENTIMENT_DL: NlpFeatureNode(A.SENTIMENT_DL, [F.SENTENCE_EMBEDDINGS], [F.DOCUMENT_CLASSIFICATION]),
        A.TRAINABLE_SENTIMENT_DL: NlpFeatureNode(A.TRAINABLE_SENTIMENT_DL, [F.SENTENCE_EMBEDDINGS],
                                                 [F.DOCUMENT_CLASSIFICATION]),

        # A.# SENTENCE_DETECTOR : NlpFeatureNode(A.SENTENCE_DETECTOR, [F.TOKEN, F.DOCUMENT], [F.DOCUMENT_CLASSIFICATION] ,
        A.STEMMER: NlpFeatureNode(A.STEMMER, [F.TOKEN], [F.TOKEN_STEMMED]),
        A.STOP_WORDS_CLEANER: NlpFeatureNode(A.STOP_WORDS_CLEANER, [F.TOKEN], [F.TOKEN_STOP_WORD_REMOVED]),
        A.SYMMETRIC_DELETE_SPELLCHECKER: NlpFeatureNode(A.SYMMETRIC_DELETE_SPELLCHECKER, [F.TOKEN],
                                                        [F.TOKEN_SPELL_CHECKED]),
        A.TEXT_MATCHER: NlpFeatureNode(A.TEXT_MATCHER, [F.DOCUMENT, F.TOKEN], [F.CHUNK]),
        A.TOKEN2CHUNK: NlpFeatureNode(A.TOKEN2CHUNK, [F.TOKEN], [F.CHUNK]),
        A.TOKEN_ASSEMBLER: NlpFeatureNode(A.TOKEN_ASSEMBLER, [F.DOCUMENT, F.TOKEN], [F.DOCUMENT]),
        A.TOKENIZER: NlpFeatureNode(A.TOKENIZER, [F.DOCUMENT], [F.TOKEN]),
        A.VIVEKN_SENTIMENT: NlpFeatureNode(A.VIVEKN_SENTIMENT, [F.TOKEN, F.DOCUMENT], [F.DOCUMENT_CLASSIFICATION]),
        A.SENTIMENT_DETECTOR: NlpFeatureNode(A.SENTIMENT_DETECTOR, [F.TOKEN, F.DOCUMENT], [F.DOCUMENT_CLASSIFICATION]),
        A.WORD_EMBEDDINGS: NlpFeatureNode(A.WORD_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.WORD_SEGMENTER: NlpFeatureNode(A.WORD_SEGMENTER, [F.DOCUMENT], [F.TOKEN]),
        A.YAKE_KEYWORD_EXTRACTION: NlpFeatureNode(A.YAKE_KEYWORD_EXTRACTION, [F.TOKEN], [F.CHUNK]),
        A.ALBERT_EMBEDDINGS: NlpFeatureNode(A.ALBERT_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.DEBERTA_FOR_TOKEN_CLASSIFICATION: NlpFeatureNode(A.DEBERTA_FOR_TOKEN_CLASSIFICATION, [F.DOCUMENT, F.TOKEN],
                                                           [F.TOKEN_CLASSIFICATION]),
        A.ALBERT_FOR_TOKEN_CLASSIFICATION: NlpFeatureNode(A.ALBERT_FOR_TOKEN_CLASSIFICATION, [F.DOCUMENT, F.TOKEN],
                                                          [F.TOKEN_CLASSIFICATION]),
        A.BERT_EMBEDDINGS: NlpFeatureNode(A.BERT_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.CAMENBERT_EMBEDDINGS: NlpFeatureNode(A.CAMENBERT_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.DEBERTA_WORD_EMBEDDINGS: NlpFeatureNode(A.BERT_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.BERT_FOR_TOKEN_CLASSIFICATION: NlpFeatureNode(A.BERT_FOR_TOKEN_CLASSIFICATION, [F.DOCUMENT, F.TOKEN],
                                                        [F.TOKEN_CLASSIFICATION]),
        A.BERT_SENTENCE_EMBEDDINGS: NlpFeatureNode(A.BERT_SENTENCE_EMBEDDINGS, [F.DOCUMENT], [F.SENTENCE_EMBEDDINGS]),
        A.DISTIL_BERT_EMBEDDINGS: NlpFeatureNode(A.DISTIL_BERT_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.DISTIL_BERT_FOR_TOKEN_CLASSIFICATION: NlpFeatureNode(A.DISTIL_BERT_FOR_TOKEN_CLASSIFICATION,
                                                               [F.DOCUMENT, F.TOKEN], [F.TOKEN_CLASSIFICATION]),
        A.ELMO_EMBEDDINGS: NlpFeatureNode(A.ELMO_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.LONGFORMER_EMBEDDINGS: NlpFeatureNode(A.LONGFORMER_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.LONGFORMER_FOR_TOKEN_CLASSIFICATION: NlpFeatureNode(A.LONGFORMER_FOR_TOKEN_CLASSIFICATION,
                                                              [F.DOCUMENT, F.TOKEN], [F.TOKEN_CLASSIFICATION]),
        A.MARIAN_TRANSFORMER: NlpFeatureNode(A.MARIAN_TRANSFORMER, [F.DOCUMENT], [F.DOCUMENT_TRANSLATED]),
        A.ROBERTA_EMBEDDINGS: NlpFeatureNode(A.ROBERTA_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.ROBERTA_FOR_TOKEN_CLASSIFICATION: NlpFeatureNode(A.ROBERTA_FOR_TOKEN_CLASSIFICATION, [F.DOCUMENT, F.TOKEN],
                                                           [F.TOKEN_CLASSIFICATION]),
        A.ROBERTA_SENTENCE_EMBEDDINGS: NlpFeatureNode(A.ROBERTA_SENTENCE_EMBEDDINGS, [F.DOCUMENT],
                                                      [F.SENTENCE_EMBEDDINGS]),
        A.T5_TRANSFORMER: NlpFeatureNode(A.T5_TRANSFORMER, [F.DOCUMENT], [F.DOCUMENT_GENERATED]),
        A.UNIVERSAL_SENTENCE_ENCODER: NlpFeatureNode(A.UNIVERSAL_SENTENCE_ENCODER, [F.DOCUMENT],
                                                     [F.SENTENCE_EMBEDDINGS]),
        A.XLM_ROBERTA_EMBEDDINGS: NlpFeatureNode(A.XLM_ROBERTA_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.XLM_ROBERTA_FOR_TOKEN_CLASSIFICATION: NlpFeatureNode(A.XLM_ROBERTA_FOR_TOKEN_CLASSIFICATION,
                                                               [F.DOCUMENT, F.TOKEN], [F.TOKEN_CLASSIFICATION]),
        A.XLM_ROBERTA_SENTENCE_EMBEDDINGS: NlpFeatureNode(A.XLM_ROBERTA_SENTENCE_EMBEDDINGS, [F.DOCUMENT],
                                                          [F.SENTENCE_EMBEDDINGS]),
        A.XLNET_EMBEDDINGS: NlpFeatureNode(A.XLNET_EMBEDDINGS, [F.DOCUMENT, F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.XLNET_FOR_TOKEN_CLASSIFICATION: NlpFeatureNode(A.XLNET_FOR_TOKEN_CLASSIFICATION, [F.DOCUMENT, F.TOKEN],
                                                         [F.TOKEN_CLASSIFICATION]),

        A.DOC2VEC: NlpFeatureNode(A.DOC2VEC, [F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.TRAIANBLE_DOC2VEC: NlpFeatureNode(A.TRAIANBLE_DOC2VEC, [F.TOKEN], [F.WORD_EMBEDDINGS]),

        A.BERT_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(A.BERT_FOR_SEQUENCE_CLASSIFICATION, [F.DOCUMENT, F.TOKEN],
                                                           [F.SEQUENCE_CLASSIFICATION]),
        A.DEBERTA_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(A.BERT_FOR_SEQUENCE_CLASSIFICATION, [F.DOCUMENT, F.TOKEN],
                                                              [F.SEQUENCE_CLASSIFICATION]),

        A.DISTIL_BERT_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(A.DISTIL_BERT_FOR_SEQUENCE_CLASSIFICATION,
                                                                  [F.DOCUMENT, F.TOKEN],
                                                                  [F.SEQUENCE_CLASSIFICATION]),

        A.XLM_ROBERTA_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(A.XLM_ROBERTA_FOR_SEQUENCE_CLASSIFICATION,
                                                                  [F.DOCUMENT, F.TOKEN],
                                                                  [F.SEQUENCE_CLASSIFICATION]),
        A.ROBERTA_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(A.ROBERTA_FOR_SEQUENCE_CLASSIFICATION,
                                                              [F.DOCUMENT, F.TOKEN],
                                                              [F.SEQUENCE_CLASSIFICATION]),
        A.LONGFORMER_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(A.LONGFORMER_FOR_SEQUENCE_CLASSIFICATION,
                                                                 [F.DOCUMENT, F.TOKEN],
                                                                 [F.SEQUENCE_CLASSIFICATION]),
        A.ALBERT_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(A.ALBERT_FOR_SEQUENCE_CLASSIFICATION,
                                                             [F.DOCUMENT, F.TOKEN],
                                                             [F.SEQUENCE_CLASSIFICATION]),
        A.XLNET_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(A.XLNET_FOR_SEQUENCE_CLASSIFICATION,
                                                            [F.DOCUMENT, F.TOKEN],
                                                            [F.SEQUENCE_CLASSIFICATION]),
        A.GPT2: NlpFeatureNode(A.GPT2, [F.DOCUMENT], [F.DOCUMENT_GENERATED]),
        A.WORD_2_VEC: NlpFeatureNode(A.WORD_2_VEC, [F.TOKEN], [F.WORD_EMBEDDINGS]),
        A.BERT_SENTENCE_CHUNK_EMBEDDINGS: NlpFeatureNode(A.BERT_SENTENCE_CHUNK_EMBEDDINGS, [F.DOCUMENT],
                                                         [F.NAMED_ENTITY_CONVERTED]),

    }


@dataclass
class OCR_FEATURE_NODES:
    """All avaiable Feature nodes in OCR
    Used to cast the pipeline dependency resolution algorithm into an abstract grpah
    """

    # Visual Document UnderstandingBINARY2IMAGE
    A = OCR_NODE_IDS
    F = OCR_FEATURES
    nodes = {
        A.VISUAL_DOCUMENT_CLASSIFIER: OcrFeatureNode(A.VISUAL_DOCUMENT_CLASSIFIER, [F.HOCR],
                                                     [F.VISUAL_CLASSIFIER_PREDICTION, F.VISUAL_CLASSIFIER_CONFIDENCE]),

        A.IMAGE2HOCR: OcrFeatureNode(A.IMAGE2HOCR, [F.OCR_IMAGE], [F.HOCR]),

        # VISUAL_DOCUMENT_NER : OcrFeatureNode(A.VISUAL_DOCUMENT_NER, [OcrFeature.HOCR, OcrFeature.FILE_PATH], [NlpFeature.NER_Annotation]), # TODO NlpFeature Space!

        # Object Detection
        A.IMAGE_HANDWRITTEN_DETECTOR: OcrFeatureNode(A.IMAGE_HANDWRITTEN_DETECTOR, [F.OCR_IMAGE, ], [F.OCR_REGION]),

        # TABLE Processors/Recognition TODO REGION::CELL>??
        A.IMAGE_TABLE_DETECTOR: OcrFeatureNode(A.IMAGE_TABLE_DETECTOR, [F.OCR_IMAGE, ], [F.OCR_TABLE]),
        # TODO REGION or TABLE??? IS IT THE SAME???
        A.IMAGE_TABLE_CELL_DETECTOR: OcrFeatureNode(A.IMAGE_TABLE_CELL_DETECTOR, [F.OCR_IMAGE, ], [F.OCR_TABLE_CELLS]),
        # TODO REGION or TABLE??? IS IT THE SAME???
        A.IMAGE_TABLE_CELL2TEXT_TABLE: OcrFeatureNode(A.IMAGE_TABLE_CELL2TEXT_TABLE, [F.OCR_IMAGE, F.OCR_TABLE_CELLS],
                                                      [F.OCR_TABLE]),
        # TODO OUPUT!! REGION or TABLE??? IS IT THE SAME???

        # TODO are POSITIOns  and REGIONS the same??? Regions is an ARRAY of PSOTISIONS. BUT is REGION::: TABLE??? Samefor CELLs
        # PDF Processing
        A.PDF2TEXT: OcrFeatureNode(A.PDF2TEXT, [F.BINARY_PDF, F.FILE_PATH], [F.TEXT, F.PAGE_NUM]),
        A.PDF2IMAGE: OcrFeatureNode(A.PDF2IMAGE, [F.BINARY_PDF, F.FILE_PATH, F.FALL_BACK], [F.OCR_IMAGE, F.PAGE_NUM]),
        A.IMAGE2PDF: OcrFeatureNode(A.IMAGE2PDF, [F.OCR_IMAGE, F.FILE_PATH], [F.BINARY_PDF]),
        A.TEXT2PDF: OcrFeatureNode(A.TEXT2PDF, [F.OCR_POSITIONS, F.OCR_IMAGE, F.OCR_TEXT, F.FILE_PATH, F.BINARY_PDF],
                                   [F.BINARY_PDF]),
        A.PDF_ASSEMBLER: OcrFeatureNode(A.PDF_ASSEMBLER, [F.BINARY_PDF_PAGE, F.FILE_PATH, F.PAGE_NUM], [F.BINARY_PDF]),
        A.PDF_DRAW_REGIONS: OcrFeatureNode(A.PDF_DRAW_REGIONS, [F.BINARY_PDF, F.FILE_PATH, F.OCR_POSITIONS],
                                           [F.BINARY_PDF]),
        A.PDF2TEXT_TABLE: OcrFeatureNode(A.PDF2TEXT_TABLE, [F.BINARY_DOCX, F.FILE_PATH, ], [F.OCR_TABLE]),

        # DOCX Processing
        A.DOC2TEXT: OcrFeatureNode(A.DOC2TEXT, [F.BINARY_DOCX, F.FILE_PATH, ], [F.TEXT, F.PAGE_NUM]),
        A.DOC2TEXT_TABLE: OcrFeatureNode(A.DOC2TEXT_TABLE, [F.BINARY_DOCX, F.FILE_PATH], [F.OCR_TABLE]),
        A.DOC2PDF: OcrFeatureNode(A.DOC2PDF, [F.BINARY_DOCX, F.FILE_PATH], [F.BINARY_PDF]),
        A.PPT2TEXT_TABLE: OcrFeatureNode(A.PPT2TEXT_TABLE, [F.BINARY_DOCX, F.FILE_PATH], [F.OCR_TABLE]),
        A.PPT2PDF: OcrFeatureNode(A.PPT2PDF, [F.BINARY_PPT, F.FILE_PATH], [F.BINARY_PDF]),

        # DICOM Processing
        A.DICOM2IMAGE: OcrFeatureNode(A.DICOM2IMAGE, [F.BINARY_DICOM, F.FILE_PATH],
                                      [F.OCR_IMAGE, F.PAGE_NUM, F.DICOM_METADATA]),
        A.IMAGE2DICOM: OcrFeatureNode(A.IMAGE2DICOM, [F.OCR_IMAGE, F.FILE_PATH, F.DICOM_METADATA], [F.BINARY_DICOM]),
        # Image Pre-Processing
        A.BINARY2IMAGE: OcrFeatureNode(A.BINARY2IMAGE, [F.BINARY_IMG, F.FILE_PATH], [F.OCR_IMAGE]),
        A.GPU_IMAGE_TRANSFORMER: OcrFeatureNode(A.GPU_IMAGE_TRANSFORMER, [F.OCR_IMAGE], [F.OCR_IMAGE]),

        A.IMAGE_BINARIZER: OcrFeatureNode(A.IMAGE_BINARIZER, [F.OCR_IMAGE], [F.OCR_IMAGE]),
        A.IMAGE_ADAPTIVE_BINARIZER: OcrFeatureNode(A.IMAGE_ADAPTIVE_BINARIZER, [F.OCR_IMAGE], [F.OCR_IMAGE]),
        A.IMAGE_ADAPTIVE_THRESHOLDING: OcrFeatureNode(A.IMAGE_ADAPTIVE_THRESHOLDING, [F.OCR_IMAGE], [F.OCR_IMAGE]),
        A.IMAGE_SCALER: OcrFeatureNode(A.IMAGE_SCALER, [F.OCR_IMAGE], [F.OCR_IMAGE]),
        A.IMAGE_ADAPTIVE_SCALER: OcrFeatureNode(A.IMAGE_ADAPTIVE_SCALER, [F.OCR_IMAGE], [F.OCR_IMAGE]),
        A.IMAGE_SKEW_CORRECTOR: OcrFeatureNode(A.IMAGE_SKEW_CORRECTOR, [F.OCR_IMAGE], [F.OCR_IMAGE]),

        # TODO THESE ALL BLOW??? Region???
        A.IMAGE_NOISE_SCORER: OcrFeatureNode(A.IMAGE_NOISE_SCORER, [F.OCR_IMAGE, F.OCR_REGION], [F.OCR_IMAGE]),
        # TODO WHAT IS REGION???? There is no schema for that
        A.IMAGE_REMOVE_OBJECTS: OcrFeatureNode(A.IMAGE_REMOVE_OBJECTS, [F.OCR_IMAGE], [F.OCR_IMAGE]),  # TODO
        A.IMAGE_MORPHOLOGY_OPERATION: OcrFeatureNode(A.IMAGE_MORPHOLOGY_OPERATION, [F.OCR_IMAGE], [F.OCR_IMAGE]),
        # TODO
        A.IMAGE_CROPPER: OcrFeatureNode(A.IMAGE_CROPPER, [F.OCR_IMAGE], [F.OCR_IMAGE]),  # TODO
        A.IMAGE2REGION: OcrFeatureNode(A.IMAGE2PDF, [F.OCR_IMAGE], [F.OCR_IMAGE]),  # TODO
        A.IMAGE_LAYOUT_ANALZYER: OcrFeatureNode(A.IMAGE_LAYOUT_ANALZYER, [F.OCR_IMAGE], [F.OCR_IMAGE]),  # TODO
        A.IMAGE_SPLIT_REGIONS: OcrFeatureNode(A.IMAGE_SPLIT_REGIONS, [F.OCR_IMAGE], [F.OCR_IMAGE]),  # TODO
        A.IMAGE_DRAW_REGIONS: OcrFeatureNode(A.IMAGE_DRAW_REGIONS, [F.OCR_IMAGE], [F.OCR_IMAGE]),  # TODO

        # Character Recognition .. TODO these should be correct but not 100% sure about the positions
        A.IMAGE2TEXT: OcrFeatureNode(A.IMAGE2TEXT, [F.OCR_IMAGE], [F.TEXT, F.OCR_POSITIONS]),
        A.IMAGE2TEXTPDF: OcrFeatureNode(A.IMAGE2TEXTPDF, [F.OCR_IMAGE, F.FILE_PATH, F.PAGE_NUM], [F.BINARY_PDF]),

        # TODO is ouput HOCR format as in HOCR_DOCUMENT_ASSAMBLER???
        A.IMAGE_BRANDS2TEXT: OcrFeatureNode(A.IMAGE_BRANDS2TEXT, [F.OCR_IMAGE], [F.OCR_POSITIONS, F.TEXT,
                                                                                 F.OCR_IMAGE]),
        # TODO what is the STRUCTURE of output image_brand ??? OCR_IE??
        A.POSITION_FINDER: OcrFeatureNode(A.POSITION_FINDER, [F.TEXT_ENTITY, F.OCR_PAGE_MATRIX], [F.OCR_POSITIONS]),
        # TODO COORDINATE::POSITION??
        ##  TODO Updates text at a position? I.e. Change the text at given corodinates BUT THEN why is output position???
        A.UPDATE_TEXT_POSITION: OcrFeatureNode(A.POSITION_FINDER, [F.OCR_POSITIONS, F.TEXT_ENTITY], [F.OCR_POSITIONS]),
        # TODO COORDINATE::POSITION??
        ## Cancer Document Test parser. Required Text of Header Field of something
        A.FOUNDATION_ONE_REPORT_PARSER: OcrFeatureNode(A.FOUNDATION_ONE_REPORT_PARSER, [F.OCR_TEXT, F.FILE_PATH],
                                                       [F.JSON_FOUNDATION_ONE_REPORT]),
        # HOCR
        A.HOCR_DOCUMENT_ASSEMBLER: OcrFeatureNode(A.HOCR_TOKENIZER, [F.HOCR], [F.TEXT_DOCUMENT]),
        A.HOCR_TOKENIZER: OcrFeatureNode(A.HOCR_TOKENIZER, [F.HOCR], [F.TEXT_DOCUMENT_TOKENIZED]),
    }


@dataclass
class NLP_HC_FEATURE_NODES():
    """All avaiable Feature nodes in NLP Healthcare
    Used to cast the pipeline dependency resolution algorithm into an abstract grpah

    """
    # Visual Document Understanding
    A = NLP_HC_NODE_IDS
    F = NLP_FEATURES
    H_F = NLP_HC_FEATURES
    # HC Feature Nodes
    nodes = {

        A.CHUNK_MAPPER_MODEL: NlpHcFeatureNode(A.CHUNK_MAPPER_MODEL, [F.NAMED_ENTITY_CONVERTED],
                                               [H_F.MAPPED_CHUNK]),

        A.ASSERTION_DL: NlpHcFeatureNode(A.ASSERTION_DL, [F.DOCUMENT, F.NAMED_ENTITY_CONVERTED, F.WORD_EMBEDDINGS],
                                         [H_F.ASSERTION]),
        A.TRAINABLE_ASSERTION_DL: NlpHcFeatureNode(A.TRAINABLE_ASSERTION_DL,
                                                   [F.DOCUMENT, F.NAMED_ENTITY_CONVERTED, F.WORD_EMBEDDINGS],
                                                   [H_F.ASSERTION]),
        A.ASSERTION_FILTERER: NlpHcFeatureNode(A.ASSERTION_FILTERER, [F.DOCUMENT, F.CHUNK, H_F.ASSERTION], [F.CHUNK]),
        A.ASSERTION_LOG_REG: NlpHcFeatureNode(A.ASSERTION_LOG_REG, [F.DOCUMENT, F.CHUNK, F.WORD_EMBEDDINGS],
                                              [H_F.ASSERTION]),
        A.TRAINABLE_ASSERTION_LOG_REG: NlpHcFeatureNode(A.TRAINABLE_ASSERTION_LOG_REG,
                                                        [F.DOCUMENT, F.CHUNK, F.WORD_EMBEDDINGS], [H_F.ASSERTION]),
        A.CHUNK2TOKEN: NlpHcFeatureNode(A.CHUNK2TOKEN, [F.CHUNK], [F.TOKEN]),
        A.CHUNK_ENTITY_RESOLVER: NlpHcFeatureNode(A.CHUNK_ENTITY_RESOLVER, [F.TOKEN, F.WORD_EMBEDDINGS],
                                                  [H_F.RESOLVED_ENTITY]),
        A.TRAINABLE_CHUNK_ENTITY_RESOLVER: NlpHcFeatureNode(A.TRAINABLE_CHUNK_ENTITY_RESOLVER,
                                                            [F.TOKEN, F.WORD_EMBEDDINGS], [H_F.RESOLVED_ENTITY]),
        A.CHUNK_FILTERER: NlpHcFeatureNode(A.CHUNK_FILTERER, [F.DOCUMENT, F.CHUNK], [F.CHUNK]),  # TODO chunk subtype?,
        A.CHUNK_KEY_PHRASE_EXTRACTION: NlpHcFeatureNode(A.CHUNK_KEY_PHRASE_EXTRACTION, [F.DOCUMENT, F.CHUNK],
                                                        [F.CHUNK]),
        A.CHUNK_MERGE: NlpHcFeatureNode(A.CHUNK_MERGE, [F.CHUNK, F.CHUNK], [F.CHUNK]),
        A.CONTEXTUAL_PARSER: NlpHcFeatureNode(A.CONTEXTUAL_PARSER, [F.DOCUMENT, F.TOKEN], [F.CHUNK]),
        A.DE_IDENTIFICATION: NlpHcFeatureNode(A.DE_IDENTIFICATION, [F.DOCUMENT, F.TOKEN, F.NAMED_ENTITY_CONVERTED],
                                              [F.DOCUMENT_DE_IDENTIFIED]),
        A.TRAINABLE_DE_IDENTIFICATION: NlpHcFeatureNode(A.DE_IDENTIFICATION, [F.DOCUMENT, F.TOKEN, F.CHUNK],
                                                        [F.DOCUMENT]),
        A.DOCUMENT_LOG_REG_CLASSIFIER: NlpHcFeatureNode(A.DOCUMENT_LOG_REG_CLASSIFIER, [F.TOKEN],
                                                        [F.DOCUMENT_CLASSIFICATION]),
        A.TRAINABLE_DOCUMENT_LOG_REG_CLASSIFIER: NlpHcFeatureNode(A.TRAINABLE_DOCUMENT_LOG_REG_CLASSIFIER, [F.TOKEN],
                                                                  [F.DOCUMENT_CLASSIFICATION]),
        A.DRUG_NORMALIZER: NlpHcFeatureNode(A.DRUG_NORMALIZER, [F.DOCUMENT], [F.DOCUMENT_NORMALIZED]),
        # A.# FEATURES_ASSEMBLER : NlpHcFeatureNode( [H_F.FEATURE_VECTOR]) # TODO data types?,
        A.GENERIC_CLASSIFIER: NlpHcFeatureNode(A.GENERIC_CLASSIFIER, [H_F.FEATURE_VECTOR], [F.DOCUMENT_CLASSIFICATION]),
        A.TRAINABLE_GENERIC_CLASSIFIER: NlpHcFeatureNode(A.TRAINABLE_GENERIC_CLASSIFIER, [H_F.FEATURE_VECTOR],
                                                         [F.DOCUMENT_CLASSIFICATION]),
        A.IOB_TAGGER: NlpHcFeatureNode(A.IOB_TAGGER, [F.TOKEN, F.CHUNK], [F.NAMED_ENTITY_IOB]),
        A.MEDICAL_NER: NlpHcFeatureNode(A.MEDICAL_NER, [F.DOCUMENT, F.TOKEN, F.WORD_EMBEDDINGS], [F.NAMED_ENTITY_IOB]),
        A.TRAINABLE_MEDICAL_NER: NlpHcFeatureNode(A.TRAINABLE_MEDICAL_NER, [F.DOCUMENT, F.TOKEN, F.WORD_EMBEDDINGS],
                                                  [F.NAMED_ENTITY_IOB]),
        A.NER_CHUNKER: NlpHcFeatureNode(A.NER_CHUNKER, [F.DOCUMENT, F.NAMED_ENTITY_IOB], [F.CHUNK]),
        A.NER_CONVERTER_INTERNAL: NlpHcFeatureNode(A.NER_CONVERTER_INTERNAL, [F.DOCUMENT, F.TOKEN, F.NAMED_ENTITY_IOB],
                                                   [F.NAMED_ENTITY_CONVERTED]),
        A.NER_DISAMBIGUATOR: NlpHcFeatureNode(A.NER_DISAMBIGUATOR, [F.CHUNK, F.SENTENCE_EMBEDDINGS],
                                              [H_F.DISAMBIGUATION]),
        A.RELATION_NER_CHUNKS_FILTERER: NlpHcFeatureNode(A.RELATION_NER_CHUNKS_FILTERER,
                                                         [F.CHUNK, F.UNLABLED_DEPENDENCY],
                                                         [F.CHUNK]),
        A.RE_IDENTIFICATION: NlpHcFeatureNode(A.RE_IDENTIFICATION, [F.DOCUMENT, F.CHUNK], [F.DOCUMENT_RE_IDENTIFIED]),
        A.RELATION_EXTRACTION: NlpHcFeatureNode(A.RELATION_EXTRACTION,
                                                [F.NAMED_ENTITY_CONVERTED, F.WORD_EMBEDDINGS, F.POS,
                                                 F.UNLABLED_DEPENDENCY],
                                                [H_F.RELATION]),

        A.ZERO_SHOT_RELATION_EXTRACTION: NlpHcFeatureNode(A.ZERO_SHOT_RELATION_EXTRACTION,
                                                          [F.NAMED_ENTITY_CONVERTED, F.DOCUMENT, ],
                                                          [H_F.RELATION]),

        A.TRAINABLE_RELATION_EXTRACTION: NlpHcFeatureNode(A.TRAINABLE_RELATION_EXTRACTION,
                                                          [F.NAMED_ENTITY_CONVERTED, F.WORD_EMBEDDINGS, F.POS,
                                                           F.UNLABLED_DEPENDENCY],
                                                          [H_F.RELATION]),
        A.RELATION_EXTRACTION_DL: NlpHcFeatureNode(A.RELATION_EXTRACTION_DL,
                                                   [F.NAMED_ENTITY_CONVERTED, F.DOCUMENT], [H_F.RELATION]),
        A.SENTENCE_ENTITY_RESOLVER: NlpHcFeatureNode(A.SENTENCE_ENTITY_RESOLVER,
                                                     [F.DOCUMENT_FROM_CHUNK, F.SENTENCE_EMBEDDINGS],
                                                     [H_F.RESOLVED_ENTITY]),
        A.TRAINABLE_SENTENCE_ENTITY_RESOLVER: NlpHcFeatureNode(A.TRAINABLE_SENTENCE_ENTITY_RESOLVER,
                                                               [F.SENTENCE_EMBEDDINGS], [H_F.ASSERTION]),
        A.MEDICAL_BERT_FOR_TOKEN_CLASSIFICATION: NlpFeatureNode(A.MEDICAL_BERT_FOR_TOKEN_CLASSIFICATION,
                                                                [F.DOCUMENT, F.TOKEN],
                                                                [F.TOKEN_CLASSIFICATION]),

        A.MEDICAL_BERT_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(A.MEDICAL_BERT_FOR_SEQUENCE_CLASSIFICATION,
                                                                   [F.DOCUMENT, F.TOKEN],
                                                                   [F.SEQUENCE_CLASSIFICATION]),
        A.MEDICAL_DISTILBERT_FOR_SEQUENCE_CLASSIFICATION: NlpFeatureNode(
            A.MEDICAL_DISTILBERT_FOR_SEQUENCE_CLASSIFICATION, [F.DOCUMENT, F.TOKEN],
            [F.SEQUENCE_CLASSIFICATION]),

    }
