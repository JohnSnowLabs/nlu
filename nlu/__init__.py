import nlu
import logging
from nlu.namespace import NameSpace
from sys import exit
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger('nlu')
# logger.setLevel(logging.INFO)
logger.setLevel(logging.CRITICAL)
ch = logging.StreamHandler()
ch.setLevel(logging.CRITICAL)

logger.addHandler(ch)

import gc
from nlu.pipeline import *

from nlu import info
from nlu.info import ComponentInfo
from nlu.components import tokenizer, stemmer, spell_checker, normalizer, lemmatizer, embeddings, chunker, embeddings_chunker
# Main components
from nlu.components.classifier import Classifier
from nlu.components.lemmatizer import Lemmatizer
from nlu.components.spell_checker import SpellChecker
from nlu.components.labeled_dependency_parser import LabeledDependencyParser as LabledDepParser
from nlu.components.unlabeled_dependency_parser import UnlabeledDependencyParser as UnlabledDepParser
from nlu.components.sentence_detector import NLUSentenceDetector

from nlu.components.dependency_untypeds.unlabeled_dependency_parser.unlabeled_dependency_parser import UnlabeledDependencyParser
from nlu.components.dependency_typeds.labeled_dependency_parser.labeled_dependency_parser import \
    LabeledDependencyParser

# 0 Base internal Spark NLP structure required for all JSL components
from nlu.components.utils.document_assembler.spark_nlp_document_assembler import SparkNlpDocumentAssembler
from nlu.components.utils.ner_to_chunk_converter.ner_to_chunk_converter import NerToChunkConverter

# we cant call the embdding file "embeddings" because namespacing wont let us import the Embeddings class inside of it then
from nlu.components.embedding import Embeddings
from nlu.components.util import Util
from nlu.components.utils.ner_to_chunk_converter import ner_to_chunk_converter

# sentence
from nlu.components.sentence_detectors.pragmatic_sentence_detector.sentence_detector import PragmaticSentenceDetector
from nlu.components.sentence_detectors.deep_sentence_detector.deep_sentence_detector import SentenDetectorDeep
# Embeddings
from nlu.components.embeddings.albert.spark_nlp_albert import SparkNLPAlbert
from nlu.components.embeddings.sentence_bert.BertSentenceEmbedding import BertSentence

from nlu.components.embeddings.bert.spark_nlp_bert import SparkNLPBert
from nlu.components.embeddings.elmo.spark_nlp_elmo import SparkNLPElmo
from nlu.components.embeddings.xlnet.spark_nlp_xlnet import SparkNLPXlnet
from nlu.components.embeddings.use.spark_nlp_use import SparkNLPUse
from nlu.components.embeddings.glove.glove import Glove

# classifiers
from nlu.components.classifiers.classifier_dl.classifier_dl import ClassifierDl
from nlu.components.classifiers.multi_classifier.multi_classifier import MultiClassifier
from nlu.components.classifiers.yake.yake import Yake
from nlu.components.classifiers.language_detector.language_detector import LanguageDetector
from nlu.components.classifiers.named_entity_recognizer_crf.ner_crf import NERDLCRF
from nlu.components.classifiers.ner.ner_dl import NERDL
from nlu.components.classifiers.sentiment_dl.sentiment_dl import SentimentDl
from nlu.components.classifiers.vivekn_sentiment.vivekn_sentiment_detector import ViveknSentiment
from nlu.components.classifiers.pos.part_of_speech_jsl import PartOfSpeechJsl

#matchers
from nlu.components.matchers.date_matcher.date_matcher import DateMatcher
from nlu.components.matchers.regex_matcher.regex_matcher import RegexMatcher
from nlu.components.matchers.text_matcher.text_matcher import TextMatcher

from  nlu.components.matcher import Matcher


# token level operators
from nlu.components.tokenizer import Tokenizer
from nlu.components.lemmatizer import Lemmatizer
from nlu.components.stemmer import Stemmer
from nlu.components.normalizer import Normalizer
from nlu.components.stopwordscleaner import StopWordsCleaner
from nlu.components.stemmers.stemmer.spark_nlp_stemmer import SparkNLPStemmer
from nlu.components.normalizers.normalizer.spark_nlp_normalizer import SparkNLPNormalizer
from nlu.components.lemmatizers.lemmatizer.spark_nlp_lemmatizer import SparkNLPLemmatizer
from nlu.components.stopwordscleaners.stopwordcleaner.nlustopwordcleaner import NLUStopWordcleaner
from nlu.components.stopwordscleaner import StopWordsCleaner
## spell
from nlu.components.spell_checkers.norvig_spell.norvig_spell_checker import NorvigSpellChecker
from nlu.components.spell_checkers.context_spell.context_spell_checker import ContextSpellChecker
from nlu.components.spell_checkers.symmetric_spell.symmetric_spell_checker import SymmetricSpellChecker
from nlu.components.tokenizers.default_tokenizer.default_tokenizer import DefaultTokenizer
from nlu.components.chunkers.default_chunker.default_chunker import DefaultChunker
from nlu.components.embeddings_chunks.chunk_embedder.chunk_embedder import ChunkEmbedder
from nlu.components.chunkers.ngram.ngram import NGram

# sentence
from nlu.components.utils.sentence_detector.sentence_detector import SparkNLPSentenceDetector

from nlu.info import AllComponentsInfo
global nlu_package_location
nlu_package_location = nlu.__file__[:-11]

global all_components_info
all_components_info = nlu.AllComponentsInfo()

global active_pipes
active_pipes = []
global spark_started, spark
spark_started = False
spark = None



def load(request, verbose = False):
    '''
    Load either a prebuild pipeline or a set of components identified by a whitespace seperated list of components
    :param request: A NLU model/pipeline/component reference
    :return: returns a fitted nlu pipeline object
    '''
    spark =  sparknlp.start()
    spark_started = True
    if verbose:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    try :         
        components_requested = request.split(' ')
        pipe = NLUPipeline()
        for nlu_reference in components_requested:
            nlu_reference.replace(' ','')
            if len(nlu_reference) <=1 : continue # white space
            # if component == ' ' : continue
            nlu_component = parse_component_data_from_name_query(nlu_reference)
            if type(nlu_component) == type([]): # if we get a list of components, then the NLU reference is a pipeline, we do not need to check order
                # lists are parsed down to multiple components
                for c in nlu_component: pipe.add(c,nlu_reference)
            else:
                pipe.add(nlu_component,nlu_reference)
                pipe = pipeline.PipelineQueryVerifier.check_and_fix_nlu_pipeline(pipe)
            
    except :
        import sys
        e = sys.exc_info()
        print(e[0])
        print(e[1])
        
        print("Something went wrong during loading and fitting the pipe. Check the other prints for more information and also verbose mode. Did you use a correct model reference?")

        return NLU_error()
    active_pipes.append(pipe)
    return pipe



def build(component_request):
    '''
    Build a nlu pipeline stack from a list of NLU components or just one NLU component
    See build_api_tests.py for usage
    :param component_request: A NLU component object or list of NLU components
    :return: a NLU pipeline which is fitted and ready to generate predictions via pipe.predict('Hello whats up')
    '''

    pipe = NLUPipeline()
    # after this, every component in the pipelCould not resolve singularine.components list should be a NLU.component type!
    if type(component_request) == str:  pass  # just one component or prebuilt pipe
    if type(component_request) == list:
        for component_to_build in component_request:
            if type(component_to_build) == str:
                pass  # string query
            else:
                pipe.add(component_to_build)  # its already built

    pipe = pipeline.PipelineQueryVerifier.check_and_fix_nlu_pipeline(pipe)
    # pipe.fit()
    return pipe



def get_default_component_of_type(missing_component_type):
    '''
    This function returns a default component for a missing component type.
    It is used to auto complete pipelines, which are missng required components.
    These represents defaults for many applications and should be set wisely.
    :param missing_component_type: String which is either just the component type or componenttype@spark_nlp_reference which stems from a models storageref and refers to some pretrained embeddings or model
    :return: a NLU component which is a either the default if there is no '@' in the @param missing_component_type or a default component for that particualar type
    '''

    logger.info('Getting default for missing_component_type=%s' , missing_component_type)
    if not '@' in missing_component_type :
        #get default models if there is no @ in the model name included
        if missing_component_type == 'document': return Util('document_assembler')
        if missing_component_type == 'sentence': return Util('sentence_detector')
        if missing_component_type == 'sentence_embeddings': return Embeddings('use')
        if 'token' in missing_component_type: return nlu.components.tokenizer.Tokenizer("default_tokenizer")
        if missing_component_type == 'word_embeddings': return Embeddings('bert')
        if missing_component_type == 'pos':   return Classifier('pos')
        if missing_component_type == 'ner':   return Classifier('named_entity_recognizer_crf')
        if missing_component_type == 'ner_converter':   return Util('ner_converter')
        if missing_component_type == 'chunk': return nlu.chunker.Chunker()
        if missing_component_type == 'ngram': return nlu.chunker.Chunker('ngram')
        if missing_component_type == 'chunk_embeddings': return embeddings_chunker.EmbeddingsChunker()
        if missing_component_type == 'unlabeled_dependency': return UnlabledDepParser()
        if missing_component_type == 'labled_dependency': return LabledDepParser('dep')
        if missing_component_type == 'date': return nlu.Matcher('date')
        if missing_component_type == 'ner_converter': return Util('ner_converter')

    else :
        #if there is an @ in the name, we must get some specific pretrained model from the sparknlp reference that should follow after the @
        missing_component_type, sparknlp_reference = missing_component_type.split('@')
        if 'embed' in missing_component_type:
            return construct_component_from_identifier(language='en', component_type=sparknlp_reference.split("_")[0], dataset='', component_embeddings='', nlu_reference='',
                                                       sparknlp_reference=sparknlp_reference)
        if 'pos' in missing_component_type or 'ner' in missing_component_type:
            return construct_component_from_identifier(language='en', component_type='classifier', dataset='', component_embeddings='', nlu_reference='',
                                                       sparknlp_reference=sparknlp_reference)
        if 'chunk_embeddings' in missing_component_type: return embeddings_chunker.EmbeddingsChunker()
        if 'unlabeled_dependency' in missing_component_type or 'dep.untyped' in missing_component_type:
            return UnlabledDepParser('unlabeled_dependency_parser') 
        if 'labled_dependency' in missing_component_type or 'dep.typed' in missing_component_type :
            return LabledDepParser('labeled_dependency_parser') 
        if 'date' in missing_component_type:

            return None

        logger.exception("Could not resolve default component type for missing type=%s", missing_component_type)


def parse_component_data_from_name_query(request, detect_lang=False):
    '''
    This method implements the main namespace for all component names. It parses the input request and passes the data to a resolver method which searches the namespace for a Component for the input request
    It returns a list of NLU.component objects or just one NLU.component object alone if just one component was specified.
    It maps a correctly namespaced name to a corrosponding component for pipeline
    If no lang is provided, default language eng is assumed.
    General format  <lang>.<class>.<dataset>.<embeddings>
    For embedding format : <lang>.<class>.<variant>
    :param request: User request (should be a NLU reference)
    :param detect_lang: Wether to automatically  detect language
    :return: Pipeline or component for the NLU reference
    '''

    infos = request.split('.')
    if len(infos) < 0: print('ERROR INVALID COMPONENT NAME')
    language = ''
    component_type = ''
    dataset = ''
    component_embeddings = ''
    component_pipe = []
    # 1. Check if either a default cmponent or one specific pretrained component or pipe  or alias of them is is requested without more sepcificatin about lang,dataset or embeding.
    # I.e. 'explain_ml' , 'explain; 'albert_xlarge_uncased' or ' tokenize'  or 'sentiment' s requested. in this case, every possible annotator must be checked.
    #format <class> or <nlu identifier>
    
    if len(infos) == 0:
        logger.exception("Split  on query is 0.")
    # Query of format <class>, no embeds,lang or dataset specified
    elif len(infos) == 1:
        logger.info('Setting default lang to english')
        language = 'en'
        if infos[0] in all_components_info.all_components or all_components_info.all_component_types:
            component_type = infos[0]
    #  check if it is any query of style #<lang>.<class>.<dataset>.<embeddings>
    elif infos[0] in all_components_info.all_languages:
        language = infos[0]
        component_type = infos[1]

        if len(infos) == 3:  # dataset specified
            dataset = infos[2]
        if len(infos) == 4:  # embeddings specified
            component_embeddings = infos[3]
    
    # passing embed_sentence can have format embed_sentence.lang.embedding or embed_sentence.embedding
    # i.e. embed_sentence.bert  
    # fr.embed_sentence.bert will automatically select french bert thus no embed_sentence.en.bert or simmilar is required
    # embed_sentence.bert or en.embed_sentence.bert
    # name does not start with a language
    # so query has format <class>.<dataset>
    elif len(infos) == 2:
        logger.info('Setting default lang to english')
        language = 'en'
        component_type = infos[0]
        dataset = infos[1]
    # query has format <class>.<dataset>.<embeddings>
    elif len(infos) == 3:
        logger.info('Setting default lang to english')
        language = 'en'
        component_type = infos[0]
        dataset = infos[1]
        component_embeddings = infos[1]

    logger.info(
        'For input query %s detected : \n lang: %s  , component type: %s , component dataset: %s , component embeddings  %s  ',
        request, language, component_type, dataset, component_embeddings)
    resolved_component = resolve_component_from_parsed_query_data(language, component_type, dataset, component_embeddings, request)

    if resolved_component == None :
        logger.exception("EXCEPTION: Could not create a component for nlu reference=%s", request)
        return None
    return resolved_component


def resolve_component_from_parsed_query_data(language, component_type, dataset, component_embeddings, full_request):
    '''
    Searches the NLU name spaces for a matching NLU reference. From that NLU reference, a SparkNLP reference will be aquired which resolved to a SparkNLP pretrained model or pipeline
    :param full_request: Full request which was passed to nlu.load()
    :param language: parsed language, may never be  '' and should be default 'en'
    :param component_type: parsed component type. may never be ''
    :param dataset: parsed dataset, can be ''
    :param component_embeddings: parsed embeddigns used for the component, can be ''
    :return: returns the nlu.Component class that corrosponds to this component. If it is a pretrained pipeline, it will return a list of components(?)
    '''
    component_kind = ''  # either model or pipe or auto_pipe
    sparknlp_reference = ''
    logger.info('Searching local Namespaces for SparkNLP reference.. ')
    resolved = False
    if resolved == False and language in NameSpace.pretrained_pipe_references.keys():
        if full_request in NameSpace.pretrained_pipe_references[language].keys():
            component_kind = 'pipe'
            sparknlp_reference = NameSpace.pretrained_pipe_references[language][full_request]
            logger.info('Found Spark NLP reference in pretrained pipelines namespace')
            resolved = True

    if resolved == False and language in NameSpace.pretrained_models_references.keys():
        if full_request in NameSpace.pretrained_models_references[language].keys():
            component_kind = 'model'
            sparknlp_reference = NameSpace.pretrained_models_references[language][full_request]
            logger.info('Found Spark NLP reference in pretrained models namespace')
            resolved = True

    if resolved == False and full_request in NameSpace.default_pretrained_component_references.keys():
        sparknlp_data = NameSpace.default_pretrained_component_references[full_request]
        component_kind = sparknlp_data[1]
        sparknlp_reference = sparknlp_data[0]
        logger.info('Found Spark NLP reference in language free aliases namespace')
        resolved = True
    if resolved == False :
        resolved = True
        component_kind = 'component'
        logger.info('Could not find reference in NLU namespace. Assuming it is a component.')
        
    if component_kind == 'pipe':
        logger.info('Inferred Spark reference for pipeline :  %s', sparknlp_reference)
        constructed_components = construct_component_from_pipe_identifier(language, sparknlp_reference)
        return constructed_components
    elif component_kind == 'model' or 'component':
        constructed_component = construct_component_from_identifier(language, component_type, dataset,
                                                                    component_embeddings, full_request,
                                                                    sparknlp_reference)
        logger.info('Inferred Spark reference for model :  %s', sparknlp_reference)
        return constructed_component
    else:
        logger.exception("EXCEPTION : Could not resolve query=%s for kind=%s and reference=%s in any of NLU's namespaces ", full_request, component_kind,
                         sparknlp_reference)
        return None


def construct_component_from_pipe_identifier(language, sparknlp_reference):
    '''
    # creates a list of components from a Spark NLP Pipeline reference
    # 1. download pipeline
    # 2. unpack pipeline to annotators and create list of nlu components
    # 3. return list of nlu components
    :param language: language of the pipeline
    :param sparknlp_reference: Reference to a spark nlp petrained pipeline
    :return: Each element of the SaprkNLP pipeline wrapped as a NLU componed inside of a list
    '''
    logger.info("Starting Spark NLP to NLU pipeline conversion process")
    from sparknlp.pretrained import PretrainedPipeline
    if 'language' in sparknlp_reference : language='xx' #special edge case for lang detectors
    pipe = PretrainedPipeline(sparknlp_reference, lang=language)
    constructed_components = []
    for component in pipe.light_model.pipeline_model.stages:
        logger.info("Extracting model from Spark NLP pipeline: %s and creating Component", component)
        parsed=''
        parsed = str(component).split('_')[0].lower()
        logger.info("Parsed Component for : %s", parsed)
        
        if 'NerConverter' in  component.name : constructed_components.append(Util(component_name='ner_converter', model=component)) 
        elif parsed == 'match': constructed_components.append(nlu.Matcher(model=component)) 
        elif parsed == 'document': constructed_components.append(nlu.Util(model=component)) 
        elif parsed == 'sentence': constructed_components.append(nlu.Util(component_name='sentence_detector',model=component)) # todo differentiate normal and deep detector
        elif parsed == 'regex': constructed_components.append(nlu.Matcher(component_name='regex', model=component))
        elif parsed == 'text': constructed_components.append(nlu.Matcher(model=component))
        elif parsed == 'spell': constructed_components.append(nlu.SpellChecker(model=component))
        elif parsed == 'lemmatizer': constructed_components.append(nlu.lemmatizer.Lemmatizer(model=component))
        elif parsed == 'normalizer': constructed_components.append(nlu.lemmatizer.Normalizer(model=component))
        elif parsed == 'stemmer': constructed_components.append(nlu.stemmer.Stemmer(model=component))
        elif parsed == 'pos' or parsed =='language': constructed_components.append(nlu.Classifier(model=component))
        elif parsed == 'word': constructed_components.append(nlu.Embeddings(model=component))
        elif parsed == 'ner' or  parsed == 'nerdlmodel': constructed_components.append(nlu.Classifier(component_name='ner',model=component))
        elif parsed == 'dependency': constructed_components.append(nlu.Util(model=component))
        elif parsed == 'typed': constructed_components.append(nlu.Util(model=component)) # todo util abuse
        elif parsed == 'multi': constructed_components.append(nlu.Util(model=component)) # todo util abuse 
        elif parsed == 'sentimentdlmodel': constructed_components.append(nlu.Classifier(model=component))
        elif parsed in ['universal','bert','albert', 'elmo', 'xlnet', 'glove','electra','covidbert','small_bert','']  : constructed_components.append(nlu.Embeddings(model=component))
        elif parsed == 'vivekn': constructed_components.append(nlu.Classifier(component_name='vivekn', model=component))
        elif parsed == 'chunker': constructed_components.append(nlu.chunker.Chunker(model=component))
        elif parsed == 'ngram': constructed_components.append(nlu.chunker.Chunker(model=component))
        elif '2e2' in parsed: constructed_components.append(nlu.Embeddings(model=component))
        elif parsed == 'embeddings_chunk': constructed_components.append(embeddings_chunker.EmbeddingsChunker(model=component))
        elif parsed == 'stopwords': constructed_components.append(nlu.StopWordsCleaner(model=component))
        
        logger.info("Extracted into NLU Component type : %s", parsed)
        if None in constructed_components :
            logger.exception("EXCEPTION: Could not infer component type for lang=%s and sparknlp_reference=%s during pipeline conversion,", language,sparknlp_reference)
            return None
    return constructed_components


def construct_component_from_identifier(language, component_type, dataset, component_embeddings, nlu_reference,
                                        sparknlp_reference):
    '''
    Creates a NLU component from a pretrained SparkNLP model reference or Class reference.
    Class references will return default pretrained models
    :param language: Language of the sparknlp model reference
    :param component_type: Class which will be used to instantiate the model
    :param dataset: Dataset that the model was trained on
    :param component_embeddings: Embedded that the models was traiend on (if any)
    :param nlu_reference: Full user request
    :param sparknlp_reference: Full Spark NLP reference
    :return: Returns a NLU component which embelished the Spark NLP pretrained model and class for that model
    '''
    logger.info('Creating singular NLU component for type=%s sparknlp reference=%s , dataset=%s, language=%s ', component_type, sparknlp_reference, dataset, language)
    try : 
        if sparknlp_reference == 'yake':
            return Classifier('yake')
        elif 'bert' in dataset or component_type == 'embed' or 'albert' in component_type or 'bert' in component_type or 'xlnet' in component_type or 'use' in component_type or 'glove' in component_type or 'elmo' in component_type or 'tfhub_use' in sparknlp_reference\
                or 'bert' in sparknlp_reference or 'labse' in sparknlp_reference or component_type =='embed_sentence' or 'electra' in nlu_reference:
            if component_type == 'embed' and dataset != '' :
                return Embeddings(component_name=dataset, language=language, get_default=False,
                                  sparknlp_reference=sparknlp_reference)
            elif component_type == 'embed' :  return Embeddings(component_name=sparknlp_reference) #default
            else : return Embeddings(component_name=component_type, language=language, get_default=False,
                              sparknlp_reference=sparknlp_reference)
        elif component_type == 'classify' or  'e2e' in sparknlp_reference:
            if component_type == 'classify' and dataset != '' :
                return Classifier(component_name=dataset, language=language, get_default=False,
                                  sparknlp_reference=sparknlp_reference)
            else : return Classifier(component_name=component_type, language=language, get_default=False,
                              sparknlp_reference=sparknlp_reference)
        elif component_type == 'tokenize':
            return nlu.tokenizer.Tokenizer(component_name=component_type, language=language, get_default=False,
                                           sparknlp_reference=sparknlp_reference)
        elif component_type == 'pos':
            return Classifier(component_name=component_type, language=language, get_default=False,
                              sparknlp_reference=sparknlp_reference)
        elif component_type == 'ner' or 'ner_dl' in sparknlp_reference:
            return Classifier(component_name='ner', language=language, get_default=False,
                              sparknlp_reference=sparknlp_reference)
        elif component_type == 'sentiment':
            return Classifier(component_name=component_type, language=language, get_default=False,
                              sparknlp_reference=sparknlp_reference)
        elif component_type == 'emotion':
            return Classifier(component_name=component_type, language=language, get_default=False,
                              sparknlp_reference=sparknlp_reference)
        elif component_type == 'spell':
            return SpellChecker(component_name=component_type, language=language, get_default=False,
                                sparknlp_reference=sparknlp_reference, dataset = dataset)
        elif component_type == 'dep' and dataset!='untyped' :# There are no trainable dep parsers this gets only default dep
            return LabledDepParser(component_name='labeled_dependency_parser', language=language, get_default=True,
                                   sparknlp_reference=sparknlp_reference)
        elif component_type == 'dep.untyped' or  dataset =='untyped': # There are no trainable dep parsers this gets only default dep
            return UnlabledDepParser(component_name='unlabeled_dependency_parser', language=language, get_default=True,
                                     sparknlp_reference=sparknlp_reference)
        elif component_type == 'lemma':
            return nlu.lemmatizer.Lemmatizer(component_name=component_type, language=language, get_default=False,
                                             sparknlp_reference=sparknlp_reference)
        elif component_type == 'norm':
            return nlu.normalizer.Normalizer(component_name='normalizer', language=language, get_default=True,
                                             sparknlp_reference=sparknlp_reference)
        elif component_type == 'clean' or component_type == 'stopwords' :
            return nlu.StopWordsCleaner( language=language, get_default=False,
                                             sparknlp_reference=sparknlp_reference)
        elif component_type == 'sentence_detector':
            return NLUSentenceDetector(component_name=component_type, language=language, get_default=True,
                              sparknlp_reference=sparknlp_reference)
        elif component_type == 'match':
            return Matcher(component_name=dataset, language=language, get_default=True,
                                       sparknlp_reference=sparknlp_reference)
        elif component_type == 'stem' or  component_type == 'stemm' or sparknlp_reference == 'stemmer' : 
            return Stemmer()
        elif component_type == 'chunk'  :return nlu.chunker.Chunker()
        elif component_type == 'ngram'  :return nlu.chunker.Chunker('ngram')
        elif component_type == 'embed_chunk': return embeddings_chunker.EmbeddingsChunker()
        elif component_type == 'regex' or sparknlp_reference =='regex_matcher' : return nlu.Matcher(component_name='regex')
        elif component_type == 'text' or sparknlp_reference =='text_matcher'  : return nlu.Matcher(component_name='text')

        logger.exception('EXCEPTION: Could not resolve singular Component for type=%s and sparknl reference=%s and nlu reference=%s', component_type, sparknlp_reference, nlu_reference)
        return None  
    except : # if reference is not in namespace and not a component it will cause a unrecoverable crash
        logger.exception('EXCEPTION: Could not resolve singular Component for type=%s and sparknl reference=%s and nlu reference=%s', component_type, sparknlp_reference, nlu_reference)
        return None
        


# Functionality discovery methods
def languages():
    ''' Print all languages which are avaiable in NLU Spark NLP pointer '''
    print('Languages available in NLU : \n ')
    for lang in  all_components_info.all_languages : print (lang)

def print_all_nlu_components_for_lang(lang='en'):
    '''Print all NLU components avialable for a language Spark NLP pointer'''
    if lang in all_components_info.all_languages :
        # print("All Pipelines for language"+ lang+ "\n"+)
        for nlu_reference in NameSpace.pretrained_pipe_references[lang] :
            print("nlu.load('"+ nlu_reference+ "') returns Spark NLP Pipeline:"+ NameSpace.pretrained_pipe_references[lang][nlu_reference])
        print("All Pipelines for language"+ lang+ "\n"+ NameSpace.pretrained_models_references[lang])
    
        for nlu_reference in NameSpace.pretrained_models_references[lang] :
            print("nlu.load('"+ nlu_reference + "') returns Spark NLP Model: "+ NameSpace.pretrained_models_references[lang][nlu_reference])

    else : print ("Language "+ lang+ " Does not exsist in NLU. Please check the docs or nlu.print_all_languages() for supported language references")


def print_all_nlu_components_for_lang(lang='en', type='classifier'):
    '''Print all NLU components avialable for a language Spark NLP pointer'''
    if lang in all_components_info.all_languages :
        # print('All Pipelines for language'+ lang+ '\n'+)
        for nlu_reference in NameSpace.pretrained_pipe_references[lang] :
            print("nlu.load('"+ nlu_reference+ "') returns Spark NLP Pipeline:"+ NameSpace.pretrained_pipe_references[lang][nlu_reference])
        print("All Pipelines for language"+ lang+ "\n"+ str(NameSpace.pretrained_models_references[lang]))

        for nlu_reference in NameSpace.pretrained_models_references[lang] :
            print("nlu.load('"+ nlu_reference + "') returns Spark NLP Model: "+ NameSpace.pretrained_models_references[lang][nlu_reference])

    else : print ("Language "+ lang+ " Does not exsist in NLU. Please check the docs or nlu.print_all_languages() for supported language references")


def print_components(lang='', action='' ):
    '''
    Print every single NLU reference for models and pipeliens and their Spark NLP pointer
    :param lang: Language requirements for the components filterd. See nlu.languages() for supported languages 
    :param type: Components that will be filterd.
    :return: None. This method will print its results.
    '''
    if lang !='' and action == '':
        print_all_nlu_components_for_lang(lang)
        return

    if lang !='' and action != '':
        print_all_model_kinds_for_action_and_lang(lang, action)
        return

    if lang =='' and action != '':
        print_all_model_kinds_for_action(action)
        return

    # Print entire Namespace below
    for nlu_reference in nlu.NameSpace.default_pretrained_component_references.keys():
        component_type =nlu.NameSpace.default_pretrained_component_references[nlu_reference][1][0],  # pipe or model
        print("nlu.load('"+ nlu_reference + "') '  returns Spark NLP "+ str(component_type)+ ': '+nlu.NameSpace.default_pretrained_component_references[nlu_reference][0] )

    for lang in nlu.NameSpace.pretrained_pipe_references.keys():
        for nlu_reference in nlu.NameSpace.pretrained_pipe_references[lang] :
            print("nlu.load('"+ nlu_reference+ "') for lang"+ lang+ " returns model Spark NLP model:"+ nlu.NameSpace.pretrained_pipe_references[lang][nlu_reference])

    
    for lang in nlu.NameSpace.pretrained_models_references.keys():
        for nlu_reference in nlu.NameSpace.pretrained_models_references[lang] :
            print("nlu.load('"+ nlu_reference+ "')' for lang"+ lang+ " returns model Spark NLP model: "+ nlu.NameSpace.pretrained_models_references[lang][nlu_reference])

def print_component_types():
    ''' Prints all unique component types in NLU'''
    types = []
    for key, val in nlu.all_components_info.all_components.items() : types.append(val.type)

    types = set(types)
    print("Provided component types in this NLU version are : ")
    for i, type in enumerate(types):
        print(i, '. ', type)
    return

def print_all_model_kinds_for_action(action):

    for lang, lang_models in nlu.NameSpace.pretrained_models_references.items():
        lang_printed=False
        for nlu_reference, nlp_reference in lang_models.items():
            ref_action  =  nlu_reference.split('.')
            if len(ref_action) > 1 : 
                ref_action=ref_action[1]

            if ref_action == action:
                if lang_printed==False :
                    print('For language <'+lang +'> NLU provides the following Models : ')
                    lang_printed=True
                print("nlu.load('"+ nlu_reference+ "') returns Spark NLP model "+ nlp_reference)
        

def print_all_model_kinds_for_action_and_lang(lang, action):
    lang_candidates = nlu.NameSpace.pretrained_models_references[lang]
    print("All NLU components for lang ", lang , " that peform action ", action)    
    for nlu_reference, nlp_reference in lang_candidates.items():
        ref_action  =  nlu_reference.split('.')
        if len(ref_action) > 1 : ref_action=ref_action[1]
        if ref_action == action:
            print("nlu.load('"+ nlu_reference+ "') returns Spark NLP model "+ nlp_reference)


class NLU_error():
    def __init__(self):
        pass
    def predict(self, text, output_level='error', positions='error', metadata='error'):
        print('The NLU components could not be properly created. Please check previous print messages and Verbose mode for further info')

    def print_info(self): print("Sorry something went wrong when building the pipeline. Please check verbose mode and your NLU reference.")
