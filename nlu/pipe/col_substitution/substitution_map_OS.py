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
    NerConverter : {
        'default': substitute_ner_converter_cols ,
    },
    MultiClassifierDLModel : {
        'default': 'TODO' ,

    },
    PerceptronModel : {
        'default': 'TODO',
    },
    ClassifierDLModel : {
        'default': substitute_classifier_dl_cols,
    },
    BertEmbeddings : {
        'default': substitute_word_embed_cols,
    },
    AlbertEmbeddings : {
        'default': substitute_word_embed_cols,
    },
    XlnetEmbeddings : {
        'default': substitute_word_embed_cols,
    },
    WordEmbeddingsModel : {
        'default': substitute_word_embed_cols,
    },
    ElmoEmbeddings : {
        'default': substitute_word_embed_cols,
    },
    BertSentenceEmbeddings : {
        'default': 'TODO',
    },
    UniversalSentenceEncoder : {
        'default': 'TODO',
    },
    SentenceEmbeddings : {
        'default': 'TODO',
    },
    Tokenizer : {
        'default': substitute_tokenizer_cols,
    },
    TokenizerModel : {
        'default': substitute_tokenizer_cols,
    },
    RegexTokenizer : {
        'default': substitute_tokenizer_cols,
    },
    DocumentAssembler : {
        'default': substitute_doc_assembler_cols,
    },
    SentenceDetectorDLModel : {
        'default': substitute_sentence_detector_dl_cols,
    },
    SentenceDetector : {
        'default': 'TODO',
    },
    ContextSpellCheckerModel : {
        'default': 'TODO',
    },
    SymmetricDeleteModel : {
        'default': 'TODO',
    },
    NorvigSweetingModel : {
        'default': 'TODO',
    },
    LemmatizerModel : {
        'default': 'TODO',
    },
    Normalizer : {
        'default': 'TODO',
    },
    NormalizerModel : {
        'default': 'TODO',
    },
    DocumentNormalizer : {
        'default':'TODO',
    },
    Stemmer : {
        'default': 'TODO',
    },
    NerDLModel : {
        'default': substitute_ner_dl_cols,
    },
    NerCrfModel : {
        'default': '',# TODO
    },
    LanguageDetectorDL : {
        'default': 'TODO',
    },
    DependencyParserModel : {
        'default': substitute_un_labled_dependency_cols,
    },
    TypedDependencyParserModel : {
        'default': substitute_labled_dependency_cols,
    },
    SentimentDLModel : {
        'default': 'TODO',
    },
    SentimentDetectorModel : {
        'default': 'TODO',
    },
    ViveknSentimentModel : {
        'default': 'TODO',
    },
    Chunker : {
        'default': 'TODO',
    },
    NGramGenerator : {
        'default': substitute_ngram_cols,
    },
    ChunkEmbeddings : {
        'default': substitute_chunk_embed_cols,
    },
    StopWordsCleaner : {
        'default': 'TODO',
    },
    TextMatcherModel : {
        'default': '',# TODO
    },
    RegexMatcherModel : {
        'default': '',# TODO
    },
    DateMatcher : {
        'default':'',# TODO
    },
    MultiDateMatcher : {
        'default': '',# TODO
    },
    T5Transformer : {
        'default': 'TODO',
    },
    MarianTransformer : {
        'default': 'TODO',
    },
    YakeModel : {
        'default': 'TODO',
    },
    WordSegmenterModel : {
        'default': 'TODO',
    },



    # approaches
    ViveknSentimentApproach    :{'default':'TODO' , 'default_full'  : default_full_config,},
    SentimentDLApproach        :{'default':'TODO' , 'default_full'  : default_full_config,},
    ClassifierDLApproach        :{'default':'TODO' , 'default_full'  : default_full_config,},
    MultiClassifierDLApproach  :{'default':'TODO', 'default_full'  : default_full_config,},
    NerDLApproach              :{'default':'TODO' , 'default_full'  : default_full_config,},
    PerceptronApproach         :{'default':'TODO' , 'default_full'  : default_full_config,},

}







