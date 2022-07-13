"""
Collection of universes shared across all libraries (NLP/HC/OCR), which are collections of atoms
"""
from nlu.universe.atoms import JslFeature


### ____ Annotator Feature Representations ____

class NLP_FEATURES(JslFeature):
    """
    NLP Features
    """
    #  STems from an nlp annotator in the NLP lib, i.e. Fnisher or so. Generates NO JSL-Annotation Schema for result df. Just 1 str per orw
    UNKOWN = JslFeature("unkown")

    DOCUMENT = JslFeature("document")
    DOCUMENT_FROM_CHUNK = JslFeature("document_from_chunk")
    DOCUMENT_DE_IDENTIFIED = JslFeature("document_de_identified")
    DOCUMENT_RE_IDENTIFIED = JslFeature("document_re_identified")
    DOCUMENT_NORMALIZED = JslFeature("document_normalized")
    DOCUMENT_TRANSLATED = JslFeature("document_translated")

    RAW_QUESTION = JslFeature("question")
    RAW_QUESTION_CONTEXT = JslFeature("context")
    DOCUMENT_QUESTION = JslFeature("document_question")
    DOCUMENT_QUESTION_CONTEXT = JslFeature("document_question_context")
    CLASSIFIED_SPAN = JslFeature("classified_span")




    # GPT, T5, X2IMG (PDF2IMG, IMG2IMG, etc..)
    DOCUMENT_GENERATED = JslFeature("document_generated")

    SENTENCE = JslFeature("sentence")
    TOKEN = JslFeature("token")
    COREF_TOKEN = JslFeature("coref_token")

    TOKEN_CHUNKED = JslFeature("token_chunked")
    TOKEN_SPELL_CHECKED = JslFeature("token_chunked")
    TOKEN_LEMATIZED = JslFeature("token_lemmatized")
    TOKEN_STEMMED = JslFeature("token_stemmed")
    TOKEN_NORMALIZED = JslFeature("token_stemmed")
    TOKEN_STOP_WORD_REMOVED = JslFeature("token_stemmed")


    WORDPIECE = JslFeature("wordpiece")
    ANY = JslFeature("any")
    ANY_FINISHED = JslFeature("any_finished")
    ANY_EMBEDDINGS = JslFeature("any_embeddings")
    FINISHED_EMBEDDINGS = JslFeature("word_embeddings")
    WORD_EMBEDDINGS = JslFeature("word_embeddings")
    CHUNK_EMBEDDINGS = JslFeature("chunk_embeddings")
    SENTENCE_EMBEDDINGS = JslFeature("sentence_embeddings")
    CATEGORY = JslFeature("category")
    DATE = JslFeature("date")
    MULTI_DOCUMENT_CLASSIFICATION = JslFeature('multi_document_classification')
    DOCUMENT_CLASSIFICATION = JslFeature('document_classification')
    TOKEN_CLASSIFICATION = JslFeature('token_classification')
    SEQUENCE_CLASSIFICATION = JslFeature('sequence_classification')

    SENTIMENT = JslFeature("sentiment")
    POS = JslFeature("pos")
    CHUNK = JslFeature("chunk")
    NAMED_ENTITY_IOB = JslFeature("named_entity_iob")
    NAMED_ENTITY_CONVERTED = JslFeature("named_entity_converted")
    NAMED_ENTITY_CONVERTED_AND_CONVERTED_TO_DOC = JslFeature("NAMED_ENTITY_CONVERTED_AND_CONVERTED_TO_DOC")
    NEGEX = JslFeature("negex")
    UNLABLED_DEPENDENCY = JslFeature("unlabeled_dependency")
    LABELED_DEPENDENCY = JslFeature("labeled_dependency")
    LANGUAGE = JslFeature("language")
    NODE = JslFeature("node")
    DUMMY = JslFeature("dummy")


class OCR_FEATURES(JslFeature):
    """
    OCR Features
    """

    BINARY_IMG = JslFeature("content")  # img -
    BINARY_PDF = JslFeature("content")  # pdf bin_pdf
    BINARY_PPT = JslFeature("bin_ppt")  # Powerpoint bin_ppt
    BINARY_PDF_PAGE = JslFeature("bin_pdf_page")  # just a page
    BINARY_DOCX = JslFeature("content")  # pdf2text - bin_docx
    BINARY_DOCX_PAGE = JslFeature("bin_docx_page")  # just a page
    BINARY_TOKEN = JslFeature("bin_token")  # img -
    BINARY_DICOM = JslFeature("bin_dicom")  # DICOM image
    DICOM_METADATA = JslFeature("dicom_metadata")  # DICOM metadata (json formatted)

    FILE_PATH = JslFeature("path")  # TODO this is externalL???
    TEXT = JslFeature("text")  # TODO should be same class as the spark NLP ones TODO EXTERNANMALLL??

    TEXT_ENTITY = JslFeature('text_entity')  # chunk/entity
    TEXT_DOCUMENT = JslFeature("text_document")  # TODO should be same class as the spark NLP ones
    TEXT_DOCUMENT_TOKENIZED = JslFeature("text_tokenized")  # TODO should be same class as the spark NLP ones
    HOCR = JslFeature("hocr")  # img -

    # All OCR_* features are structs generated from OCR lib
    FALL_BACK = JslFeature("fall_back")  #
    OCR_IMAGE = JslFeature("ocr_image")  # OCR struct image representation
    OCR_PAGE_MATRIX = JslFeature("ocr_page_matrix")  # OCR struct image representation
    OCR_POSITIONS = JslFeature(
        "ocr_positions")  # OCR struct POSITION representation # TODO is POSITIONS==COORDINATES???
    OCR_REGION = JslFeature("ocr_region")  # OCR array of POSITION struct
    OCR_TEXT = JslFeature("ocr_text")  # raw text extracted by OCR anno like PDFtoImage
    OCR_TABLE = JslFeature("ocr_table")  # OCR extracted table TODO array of COORDINATES/POSITION?
    OCR_TABLE_CELLS = JslFeature("ocr_table_cells")  # OCR extracted table  TODO array of COORDINATES/POSITION??
    OCR_MAPPING = JslFeature("ocr_table")  # TODO wat is MPAPING???
    PAGE_NUM = JslFeature("page_num")  # TODO is this just int or some struct?

    JSON_FOUNDATION_ONE_REPORT = JslFeature("json_foundation_one_report")

    PREDICTION_TEXT_TABLE = JslFeature("prediction_text_lable")  # TODO is this just int or some struct?
    PREDICTION_CONFIDENCE = JslFeature("prediction_confidence")  # TODO is this just int or some struct?
    VISUAL_CLASSIFIER_CONFIDENCE = JslFeature("visual_classifier_confidence")
    VISUAL_CLASSIFIER_PREDICTION = JslFeature("visual_classifier_prediction")


class NLP_HC_FEATURES(JslFeature):
    """
    NLP HC Feature aka Annotator Types
    """
    ASSERTION = JslFeature('assertion')
    RESOLVED_ENTITY = JslFeature('resolved_entity')
    FEATURE_VECTOR = JslFeature('feature_vector')
    MAPPED_CHUNK = JslFeature('mapped_chunk')
    DISAMBIGUATION = JslFeature('disambiguation')
    RELATION = JslFeature('relation')