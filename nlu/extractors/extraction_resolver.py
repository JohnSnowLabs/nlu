"""
Resolve Annotator Classes in the Pipeline to Extractor Configs and Methods

Every Annotator should have 2 configs. Some might offor multuple configs/method pairs, based on model/NLP reference.
- default/minimalistic -> Just the results of the annotations, no confidences or extra metadata
- with meta            -> A config that leverages white/black list and gets the most relevant metadata
- with positions       -> With Begins/Ends


"""
from sparknlp.annotator import *

annotator_levels_approach_based = {
    'document': [DocumentAssembler, Chunk2Doc,
                 YakeModel,
                 ],
    'sentence': [SentenceDetector, SentenceDetectorDLApproach, ],
    'chunk': [Chunker, ChunkEmbeddings,  ChunkTokenizer, Token2Chunk, TokenAssembler,
              NerConverter, Doc2Chunk,NGramGenerator],
    'token': [ NerCrfApproach, NerDLApproach,
               PerceptronApproach,
               Stemmer,
               ContextSpellCheckerApproach,
               nlu.WordSegmenter,
               Lemmatizer, TypedDependencyParserApproach, DependencyParserApproach,
               Tokenizer, RegexTokenizer, RecursiveTokenizer
        ,StopWordsCleaner, DateMatcher, TextMatcher, BigTextMatcher, MultiDateMatcher,
               WordSegmenterApproach
               ],
    # sub token is when annotator is token based but some tokens may be missing since dropped/cleanes
    # are matchers chunk or sub token?
    # 'sub_token': [StopWordsCleaner, DateMatcher, TextMatcher, BigTextMatcher, MultiDateMatcher],
    # these can be document or sentence
    'input_dependent': [ViveknSentimentApproach, SentimentDLApproach, ClassifierDLApproach,
                        LanguageDetectorDL,
                        MultiClassifierDLApproach,  SentenceEmbeddings, NorvigSweetingApproach,
                        ],

    # 'unclassified': [Yake, Ngram]
}


annotator_levels_model_based = {
    'document': [],
    'sentence': [SentenceDetectorDLModel, ],
    'chunk': [ChunkTokenizerModel, ChunkTokenizerModel, ],
    'token': [ContextSpellCheckerModel, AlbertEmbeddings, BertEmbeddings, ElmoEmbeddings, WordEmbeddings,
              XlnetEmbeddings, WordEmbeddingsModel,
              NerDLModel, NerCrfModel, PerceptronModel, SymmetricDeleteModel, NorvigSweetingModel,
              ContextSpellCheckerModel,
              TypedDependencyParserModel, DependencyParserModel,
              RecursiveTokenizerModel,
              TextMatcherModel, BigTextMatcherModel, RegexMatcherModel,
              WordSegmenterModel
              ],
    # 'sub_token': [TextMatcherModel, BigTextMatcherModel, RegexMatcherModel, ],
    'input_dependent': [BertSentenceEmbeddings, UniversalSentenceEncoder, ViveknSentimentModel,
                        SentimentDLModel, MultiClassifierDLModel, MultiClassifierDLModel, ClassifierDLModel,
                        MarianTransformer,T5Transformer

                        ],
}

all_embeddings = {
    'token' : [AlbertEmbeddings, BertEmbeddings, ElmoEmbeddings, WordEmbeddings,
               XlnetEmbeddings,WordEmbeddingsModel],
    'input_dependent' : [SentenceEmbeddings, UniversalSentenceEncoder,BertSentenceEmbeddings]

}