"""
Resolve Annotator Classes in the Pipeline to Extractor Configs and Methods.
Each Spark NLP Annotator Class is mapped to at least one

Every Annotator should have 2 configs. Some might offor multuple configs/method pairs, based on model/NLP reference.
- default/minimalistic -> Just the results of the annotations, no confidences or extra metadata
- with meta            -> A config that leverages white/black list and gets the most relevant metadata
- with positions       -> With Begins/Ends


"""
from sparknlp.annotator import *
from sparknlp.base import *
from nlu.pipe.extractors.extractor_configs_open_source import *
from nlu.pipe.col_substitution.col_substitution_OS import *

OS_anno2substitution_fn = {
    NerConverter: {
        'default': substitute_ner_converter_cols,
    },
    PerceptronModel: {
        'default': substitute_pos_cols,
    },
    BertEmbeddings: {
        'default': substitute_word_embed_cols,
    },
    AlbertEmbeddings: {
        'default': substitute_word_embed_cols,
    },
    XlnetEmbeddings: {
        'default': substitute_word_embed_cols,
    },

    DistilBertEmbeddings: {
        'default': substitute_word_embed_cols,
    },
    RoBertaEmbeddings: {
        'default': substitute_word_embed_cols,
    },
    XlmRoBertaEmbeddings: {
        'default': substitute_word_embed_cols,
    },

    WordEmbeddingsModel: {
        'default': substitute_word_embed_cols,
    },
    ElmoEmbeddings: {
        'default': substitute_word_embed_cols,
    },
    BertSentenceEmbeddings: {
        'default': substitute_sent_embed_cols,
    },
    UniversalSentenceEncoder: {
        'default': substitute_sent_embed_cols,
    },
    SentenceEmbeddings: {
        'default': substitute_sent_embed_cols,
    },
    Tokenizer: {
        'default': substitute_tokenizer_cols,
    },
    TokenizerModel: {
        'default': substitute_tokenizer_cols,
    },
    RegexTokenizer: {
        'default': substitute_tokenizer_cols,
    },
    DocumentAssembler: {
        'default': substitute_doc_assembler_cols,
    },
    SentenceDetectorDLModel: {
        'default': substitute_sentence_detector_dl_cols,
    },
    SentenceDetector: {
        'default': substitute_sentence_detector_pragmatic_cols,
    },
    ContextSpellCheckerModel: {
        'default': substitute_spell_context_cols,
    },
    SymmetricDeleteModel: {
        'default': substitute_spell_symm_cols,
    },
    NorvigSweetingModel: {
        'default': substitute_spell_norvig_cols,
    },
    LemmatizerModel: {
        'default': substitute_lem_cols,
    },
    Normalizer: {
        'default': substitute_norm_cols,
    },
    NormalizerModel: {
        'default': substitute_norm_cols,
    },
    DocumentNormalizer: {
        'default': substitute_doc_norm_cols,
    },
    Stemmer: {
        'default': substitute_stem_cols,
    },
    NerDLModel: {
        'default': substitute_ner_dl_cols,
    },
    NerCrfModel: {
        'default': 'TODO',
    },
    LanguageDetectorDL: {
        'default': 'TODO',
    },
    DependencyParserModel: {
        'default': substitute_un_labled_dependency_cols,
    },
    TypedDependencyParserModel: {
        'default': substitute_labled_dependency_cols,
    },
    SentimentDLModel: {
        'default': substitute_sentiment_dl_cols,
    },
    SentimentDetectorModel: {
        'default': substitute_sentiment_dl_cols,
    },
    ViveknSentimentModel: {
        'default': substitute_sentiment_vivk_cols,
    },
    MultiClassifierDLModel: {
        'default': substitute_multi_classifier_dl_cols,
    },
    ClassifierDLModel: {
        'default': substitute_classifier_dl_cols,
    },

    Chunker: {
        'default': substitute_chunk_cols,
    },
    NGramGenerator: {
        'default': substitute_ngram_cols,
    },
    ChunkEmbeddings: {
        'default': substitute_chunk_embed_cols,
    },
    StopWordsCleaner: {
        'default': substitute_stopwords_cols,
    },



    TextMatcherModel: {
        'default': substitute_text_match_cols,
    },
    RegexMatcherModel: {
        'default': substitute_regex_match_cols,
    },
    DateMatcher: {
        'default': substitute_date_match_cols,
    },
    MultiDateMatcher: {
        'default': '',  # TODO
    },

    Doc2Chunk: {
        'default': substitute_doc2chunk_cols,
    },


    T5Transformer: {
        'default': substitute_T5_cols,
    },
    MarianTransformer: {
        'default': substitute_marian_cols,
    },
    YakeModel: {
        'default': substitute_YAKE_cols,
    },
    WordSegmenterModel: {
        'default': substitute_word_seg_cols,
    },

    # approaches
    ViveknSentimentApproach: {'default': substitute_sentiment_vivk_approach_cols,
                              'default_full': default_full_config, },
    SentimentDLApproach: {'default': substitute_sentiment_dl_approach_cols, 'default_full': default_full_config, },
    ClassifierDLApproach: {'default': substitute_classifier_dl_approach_cols, 'default_full': default_full_config, },
    MultiClassifierDLApproach: {'default': substitute_multi_classifier_dl_approach_cols,
                                'default_full': default_full_config, },
    NerDLApproach: {'default': substitute_ner_dl_approach_cols, 'default_full': default_full_config, },
    PerceptronApproach: {'default': substitute_pos_approach_cols, 'default_full': default_full_config, },

}
