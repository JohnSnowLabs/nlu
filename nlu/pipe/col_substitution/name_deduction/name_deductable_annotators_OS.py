"""Deduct name for annotator"""
"""
Resolve Annotator Classes in the Pipeline to Extractor Configs and Methods.
Each Spark NLP Annotator Class is mapped to at least one

Every Annotator should have 2 configs. Some might offor multuple configs/method pairs, based on model_anno_obj/NLP reference.
- default/minimalistic -> Just the results of the annotations, no confidences or extra metadata
- with meta            -> A config that leverages white/black list and gets the most relevant metadata
- with positions       -> With Begins/Ends


"""
from sparknlp.annotator import *
from sparknlp.base import *
name_deductable_OS = [
    NerConverter,
    BertEmbeddings,
    AlbertEmbeddings,
    XlnetEmbeddings ,
    WordEmbeddingsModel ,
    ElmoEmbeddings ,
    BertSentenceEmbeddings,
    UniversalSentenceEncoder,
    SentenceEmbeddings,
    ContextSpellCheckerModel ,
    SymmetricDeleteModel ,
    NorvigSweetingModel ,
    NerDLModel ,
    NerCrfModel,
    LanguageDetectorDL ,
    SentimentDLModel ,
    SentimentDetectorModel ,
    ViveknSentimentModel ,
    MultiClassifierDLModel,
    ClassifierDLModel ,
    ChunkEmbeddings ,
    TextMatcherModel,
    RegexMatcherModel,
    DateMatcher,
    MultiDateMatcher,
    T5Transformer,
    MarianTransformer,
    WordSegmenterModel,


    DistilBertEmbeddings,
    RoBertaEmbeddings,
    XlmRoBertaEmbeddings,

    DistilBertForTokenClassification,
    BertForTokenClassification,
    LongformerEmbeddings,
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    # approaches
    ViveknSentimentApproach    ,
    SentimentDLApproach        ,
    ClassifierDLApproach       ,
    MultiClassifierDLApproach  ,
    NerDLApproach              ,
    PerceptronApproach         ,
    Doc2Chunk,
    Chunk2Doc,
    DeBertaEmbeddings,
    # MultiDocumentAssembler,
    AlbertForQuestionAnswering,
    BertForQuestionAnswering,
    DeBertaForQuestionAnswering,
    DistilBertForQuestionAnswering,
    LongformerForQuestionAnswering,
    RoBertaForQuestionAnswering,
    XlmRoBertaForQuestionAnswering,
    # SpanBertCorefModel,
#


]


always_name_deductable_OS = [
    BertEmbeddings,
    AlbertEmbeddings,
    XlnetEmbeddings ,
    WordEmbeddingsModel ,
    ElmoEmbeddings ,
    BertSentenceEmbeddings,
    UniversalSentenceEncoder,
    SentenceEmbeddings,
    MultiClassifierDLModel,
    ClassifierDLModel ,
    ChunkEmbeddings ,
    TextMatcherModel,
    RegexMatcherModel,
    DateMatcher,
    MultiDateMatcher,
    # T5Transformer,
    # MarianTransformer,
    # WordSegmenterModel,


    DistilBertEmbeddings,
    RoBertaEmbeddings,
    XlmRoBertaEmbeddings,

    Chunk2Doc,
    DeBertaEmbeddings,

    # MultiDocumentAssembler,
    # AlbertForQuestionAnswering,
    # BertForQuestionAnswering,
    # DeBertaForQuestionAnswering,
    # DistilBertForQuestionAnswering,
    # LongformerForQuestionAnswering,
    # RoBertaForQuestionAnswering,
    # XlmRoBertaForQuestionAnswering,
    # # SpanBertCorefModel,

]












