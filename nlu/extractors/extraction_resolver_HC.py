"""
Resolve Annotator Classes in the Pipeline to Extractor Configs and Methods

Every Annotator should have 2 configs. Some might offor multuple configs/method pairs, based on model/NLP reference.
- default/minimalistic -> Just the results of the annotations, no confidences or extra metadata
- with meta            -> A config that leverages white/black list and gets the most relevant metadata
- with positions       -> With Begins/Ends
- with sentence references -> Reeturn the sentence/chunk no. reference from the metadata.
                                If a document has multi-sentences, this will map a label back to a corrosponding sentence

"""
from sparknlp.annotator import *
from nlu.extractors.extractor_configs import *
from nlu.extractors.extractor_base_data_classes import *
d = {
    NerConverter : {
        'default': default_NER_config ,
    },
    MultiClassifierDLModel : {
        'default': '',# TODO
    },
    PerceptronModel : {
        'default': default_POS_config,
    },
    # ClassifierDl : {
    #     'default': '',# TODO
    # },
    ClassifierDLModel : {
        'default': '',# TODO
    },
    BertEmbeddings : {
        'default': default_word_embedding_config,
    },
    AlbertEmbeddings : {
        'default': default_word_embedding_config,
    },
    XlnetEmbeddings : {
        'default': default_word_embedding_config,
    },
    WordEmbeddingsModel : {
        'default': default_word_embedding_config,
    },
    ElmoEmbeddings : {
        'default': default_word_embedding_config,
    },
    BertSentenceEmbeddings : {
        'default': default_sentence_embedding_config,
    },
    UniversalSentenceEncoder : {
        'default': default_sentence_embedding_config,
    },
    TokenizerModel : {
        'default': default_tokenizer_config,
    },
    DocumentAssembler : {
        'default': default_document_config,
    },
    SentenceDetectorDLModel : {
        'default': default_sentence_detector_DL_config,
    },
    SentenceDetector : {
        'default': '',# TODO
    },
    ContextSpellCheckerModel : {
        'default': '',# TODO
    },
    SymmetricDeleteModel : {
        'default': '',# TODO
    },
    NorvigSweetingModel : {
        'default': '',# TODO
    },
    LemmatizerModel : {
        'default': '',# TODO
    },
    NormalizerModel : {
        'default': '',# TODO
    },
    DocumentNormalizer : {
        'default' # TODO
    }
    ,
    Stemmer : {
        'default': default_stemm_config,
    },
    NerDLModel : {
        'default':default_NER_config, # TODO
    },
    NerCrfModel : {
        'default': '',# TODO
    },
    LanguageDetectorDL : {
        'default': '',# TODO
    },
    DependencyParserModel : {
        'default': '',# TODO
    },
    TypedDependencyParserModel : {
        'default': '',# TODO
    },
    SentimentDLModel : {
        'default': '',# TODO
    },
    SentimentDetectorModel : {
        'default': '',# TODO
    },
    ViveknSentimentModel : {
        'default': '',# TODO
    },
    Chunker : {
        'default': '',# TODO
    },
    NGramGenerator : {
        'default': '',# TODO
    },
    ChunkEmbeddings : {
        'default': '',# TODO
    },
    StopWordsCleaner : {
        'default': default_stopwords_config,# TODO
    },
    TextMatcherModel : {
        'default': '',# TODO
    },
    RegexMatcherModel : {
        'default': '',# TODO
    },
    # DateMatcher : {
    #     'default': default_'',# TODO
    # },
    MultiDateMatcher : {
        'default': '',# TODO
    },
    T5Transformer : {
        'default': '',# TODO
    },
    MarianTransformer : {
        'default': '',# TODO
    }
    PretrainedPipeline : {
        'default' : '', #TODO RLY?
    }
}







