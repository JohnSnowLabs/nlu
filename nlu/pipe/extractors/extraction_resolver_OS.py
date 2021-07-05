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

OS_anno2config = {
    NerConverter : {
        'default': default_ner_converter_config ,
        'default_full'  : default_full_config,
    },
    MultiClassifierDLModel : {
        'default': default_multi_classifier_dl_config ,
        'default_full'  : default_full_config,

    },
    PerceptronModel : {
        'default': default_POS_config,
        'default_full'  : default_full_config,
    },
    ClassifierDLModel : {
        'default': default_classifier_dl_config,
        'default_full'  : default_full_config,
    },
    BertEmbeddings : {
        'default': default_word_embedding_config,
        'default_full'  : default_full_config,
    },

    AlbertEmbeddings : {
        'default': default_word_embedding_config,
        'default_full'  : default_full_config,
    },
    XlnetEmbeddings : {
        'default': default_word_embedding_config,
        'default_full'  : default_full_config,
    },
    RoBertaEmbeddings : {
        'default': default_word_embedding_config,
        'default_full'  : default_full_config,
    },
    XlmRoBertaEmbeddings : {
        'default': default_word_embedding_config,
        'default_full'  : default_full_config,
    },
    DistilBertEmbeddings : {
        'default': default_word_embedding_config,
        'default_full'  : default_full_config,
    },
    WordEmbeddingsModel : {
        'default': default_word_embedding_config,
        'default_full'  : default_full_config,
    },
    ElmoEmbeddings : {
        'default': default_word_embedding_config,
        'default_full'  : default_full_config,
    },
    BertSentenceEmbeddings : {
        'default': default_sentence_embedding_config,
        'default_full'  : default_full_config,
    },
    UniversalSentenceEncoder : {
        'default': default_sentence_embedding_config,
        'default_full'  : default_full_config,
    },
    SentenceEmbeddings : {
        'default': default_sentence_embedding_config,
        'default_full'  : default_full_config,
    },
    Tokenizer : {
        'default': default_tokenizer_config,
        'default_full'  : default_full_config,
    },
    TokenizerModel : {
        'default': default_tokenizer_config,
        'default_full'  : default_full_config,
    },
    RegexTokenizer : {
        'default': default_tokenizer_config,
        'default_full'  : default_full_config,
    },
    DocumentAssembler : {
        'default': default_document_config,
        'default_full'  : default_full_config,
    },
    SentenceDetectorDLModel : {
        'default': default_sentence_detector_DL_config,
        'default_full'  : default_full_config,
    },
    SentenceDetector : {
        'default': default_sentence_detector_config,
        'default_full'  : default_full_config,
    },
    ContextSpellCheckerModel : {
        'default': default_spell_context_config,
        'default_full'  : default_full_config,
    },
    SymmetricDeleteModel : {
        'default': default_spell_symmetric_config,
        'default_full'  : default_full_config,
    },
    NorvigSweetingModel : {
        'default': default_spell_norvig_config,
        'default_full'  : default_full_config,
    },
    LemmatizerModel : {
        'default': default_lemma_config,
        'default_full'  : default_full_config,
    },
    Normalizer : {
        'default': default_norm_config,
        'default_full'  : default_full_config,
    },
    NormalizerModel : {
        'default': default_norm_config,
        'default_full'  : default_full_config,
    },
    DocumentNormalizer : {
        'default':default_norm_document_config,
        'default_full'  : default_full_config,
    },
    Stemmer : {
        'default': default_stemm_config,
        'default_full'  : default_full_config,
    },
    NerDLModel : {
        'default': default_NER_config,
        'meta': meta_NER_config,
        'default_full'  : default_full_config,
    },
    NerCrfModel : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },
    LanguageDetectorDL : {
        'default': default_lang_classifier_config,
        'default_full'  : default_full_config,
    },
    DependencyParserModel : {
        'default': default_dep_untyped_config,
        'default_full'  : default_full_config,
    },
    TypedDependencyParserModel : {
        'default': default_dep_typed_config,
        'default_full'  : default_full_config,
    },
    SentimentDLModel : {
        'default': default_sentiment_dl_config,
        'default_full'  : default_full_config,
    },
    SentimentDetectorModel : {
        'default': default_sentiment_config,
        'default_full'  : default_full_config,
    },
    ViveknSentimentModel : {
        'default': default_sentiment_vivk_config,
        'default_full'  : default_full_config,
    },
    Chunker : {
        'default': default_chunk_config,
        'default_full'  : default_full_config,
    },
    NGramGenerator : {
        'default': default_ngram_config,
        'default_full'  : default_full_config,
    },
    ChunkEmbeddings : {
        'default': default_chunk_embedding_config,
        'default_full'  : default_full_config,
    },
    StopWordsCleaner : {
        'default': default_stopwords_config,
        'default_full'  : default_full_config,
    },
    TextMatcherModel : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },
    TextMatcher : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },

    RegexMatcherModel : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },
    RegexMatcher : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },
    DateMatcher : {
        'default':'',# TODO
        'default_full'  : default_full_config,
    },

    MultiDateMatcher : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },

    Doc2Chunk : {
        'default': default_doc2chunk_config,
        'default_full'  : default_full_config,
    },


    T5Transformer : {
        'default': default_T5_config,
        'default_full'  : default_full_config,
    },
    MarianTransformer : {
        'default': default_marian_config,
        'default_full'  : default_full_config,
    },
    YakeModel : {
        'default': default_yake_config,
        'default_full'  : default_full_config,
    },
    WordSegmenterModel : {
        'default': default_word_segmenter_config,
        'default_full'  : default_full_config,
    },



    # approaches
    ViveknSentimentApproach    :{'default':'' , 'default_full'  : default_full_config,},
    SentimentDLApproach        :{'default':default_sentiment_dl_config , 'default_full'  : default_full_config,},
    ClassifierDLApproach        :{'default':default_classifier_dl_config , 'default_full'  : default_full_config,},
    MultiClassifierDLApproach  :{'default':default_multi_classifier_dl_config, 'default_full'  : default_full_config,},
    NerDLApproach              :{'default':default_NER_config , 'default_full'  : default_full_config,},
    PerceptronApproach         :{'default':default_POS_config , 'default_full'  : default_full_config,},


    # PretrainedPipeline : {
    #     'default' : '',
    # }
}







