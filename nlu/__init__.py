__version__ = '1.1.1'
#
# from nlu.component_resolution import parse_language_from_nlu_ref, nlu_ref_to_component, \
#     construct_component_from_pipe_identifier
import sys

def check_pyspark_install():
    try :
        from pyspark.sql import SparkSession
        try :
            import sparknlp
            v = sparknlp.start().version
            spark_major = int(v.split('.')[0])
            if spark_major >= 3 :
                raise Exception()
        except :
            print(f"Detected pyspark version={v} Which is >=3.X\nPlease run '!pip install pyspark==2.4.7' or install any pyspark>=2.4.0 and pyspark<3")
            # print(f"Or set nlu.load(version_checks=False). We disadvise from doing so, until Pyspark >=3 is officially supported in 2021.")
            return False
    except :
        print("No Pyspark installed!\nPlease run '!pip install pyspark==2.4.7' or install any pyspark>=2.4.0 with pyspark<3")
        return False
    return True

def check_python_version():
    if float(sys.version[:3]) >= 3.8:
        print("Please use a Python version with version number SMALLER than 3.8")
        print("Python versions equal or higher 3.8 are currently NOT SUPPORTED by NLU")
        return False
    return True


# if not check_pyspark_install(): raise Exception()
if not check_python_version(): raise Exception()



import nlu
import logging
from nlu.namespace import NameSpace
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

# NLU Healthcare components
from nlu.components.assertion import Asserter



from nlu import info
from nlu.info import ComponentInfo
from nlu.components import tokenizer, stemmer, spell_checker, normalizer, lemmatizer, embeddings, chunker, \
    embeddings_chunker
# Main components
from nlu.components.classifier import Classifier
from nlu.components.lemmatizer import Lemmatizer
from nlu.components.spell_checker import SpellChecker
from nlu.components.labeled_dependency_parser import LabeledDependencyParser as LabledDepParser
from nlu.components.unlabeled_dependency_parser import UnlabeledDependencyParser as UnlabledDepParser
from nlu.components.sentence_detector import NLUSentenceDetector

from nlu.components.dependency_untypeds.unlabeled_dependency_parser.unlabeled_dependency_parser import \
    UnlabeledDependencyParser
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
from nlu.components.sentence_detectors.deep_sentence_detector.deep_sentence_detector import SentenceDetectorDeep
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

# matchers
from nlu.components.matchers.date_matcher.date_matcher import DateMatcher
from nlu.components.matchers.regex_matcher.regex_matcher import RegexMatcher
from nlu.components.matchers.text_matcher.text_matcher import TextMatcher

from nlu.components.matcher import Matcher

# token level operators
from nlu.components.tokenizer import Tokenizer
from nlu.components.lemmatizer import Lemmatizer
from nlu.components.stemmer import Stemmer
from nlu.components.normalizer import Normalizer
from nlu.components.stopwordscleaner import StopWordsCleaner
from nlu.components.stemmers.stemmer.spark_nlp_stemmer import SparkNLPStemmer
from nlu.components.normalizers.normalizer.spark_nlp_normalizer import SparkNLPNormalizer
from nlu.components.normalizers.document_normalizer.spark_nlp_normalizer import SparkNLPDocumentNormalizer

from nlu.components.lemmatizers.lemmatizer.spark_nlp_lemmatizer import SparkNLPLemmatizer
from nlu.components.stopwordscleaners.stopwordcleaner.nlustopwordcleaner import NLUStopWordcleaner
from nlu.components.stopwordscleaner import StopWordsCleaner
## spell
from nlu.components.spell_checkers.norvig_spell.norvig_spell_checker import NorvigSpellChecker
from nlu.components.spell_checkers.context_spell.context_spell_checker import ContextSpellChecker
from nlu.components.spell_checkers.symmetric_spell.symmetric_spell_checker import SymmetricSpellChecker
from nlu.components.tokenizers.default_tokenizer.default_tokenizer import DefaultTokenizer
from nlu.components.tokenizers.word_segmenter.word_segmenter import WordSegmenter

from nlu.components.chunkers.default_chunker.default_chunker import DefaultChunker
from nlu.components.embeddings_chunks.chunk_embedder.chunk_embedder import ChunkEmbedder
from nlu.components.chunkers.ngram.ngram import NGram

# sentence
from nlu.components.utils.sentence_detector.sentence_detector import SparkNLPSentenceDetector
from nlu.components.sentence_detector import NLUSentenceDetector

#seq2seq
from nlu.components.seq2seqs.marian.marian import Marian
from nlu.components.seq2seqs.t5.t5 import T5
from nlu.components.sequence2sequence import Seq2Seq

from nlu.pipeline_logic import PipelineQueryVerifier

from nlu.component_resolution import *
global spark, active_pipes, all_components_info, nlu_package_location,authorized
is_authenticated=False
nlu_package_location = nlu.__file__[:-11]

active_pipes = []
spark_started = False
spark = None
authenticated = False
from nlu.info import AllComponentsInfo
from nlu.discovery import Discoverer
all_components_info = nlu.AllComponentsInfo()
discoverer = nlu.Discoverer()

import os
import sparknlp

def install_and_import_healthcare(JSL_SECRET):
    # pip install  spark-nlp-jsl==2.7.3 --extra-index-url https://pypi.johnsnowlabs.com/2.7.3-3f5059a2258ea6585a0bd745ca84dac427bca70c --upgrade
    """ Install Spark-NLP-Healthcare PyPI Package in current enviroment if it cannot be imported and liscense provided"""
    import importlib
    try:
        importlib.import_module('sparknlp_jsl')
    except ImportError:
        import pip
        # Todo update to latest version of spark-nlp-jsl
        print("Spark NLP Healthcare could not be imported. Installing latest spark-nlp-jsl PyPI package via pip...")
        pip.main(['install', 'spark-nlp-jsl==2.7.3', '--extra-index-url', f'https://pypi.johnsnowlabs.com/{JSL_SECRET}'])
    finally:
        import site
        from importlib import reload
        reload(site)
        globals()['sparknlp_jsl'] = importlib.import_module('sparknlp_jsl')

def authenticate_enviroment(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY):
    """Set Secret environ variables for Spark Context"""
    import os
    os.environ['SPARK_NLP_LICENSE'] = SPARK_NLP_LICENSE
    os.environ['AWS_ACCESS_KEY_ID']= AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

def get_authenticated_spark(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET):
    """
    Authenticates enviroment if not already done so and returns Spark Context with Healthcare Jar loaded
    0. If no Spark-NLP-Healthcare, install it via PyPi
    1. If not auth, run authenticate_enviroment()

    """
    install_and_import_healthcare(JSL_SECRET)
    authenticate_enviroment(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)

    import sparknlp_jsl
    return sparknlp_jsl.start(JSL_SECRET)

def is_authorized_enviroment():
    """ TODO"""
    global is_authenticated
    return is_authenticated

def auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET):
    """ Authenticate enviroment for JSL Liscensed models. Installs NLP-Healthcare if not in enviroment detected"""
    global is_authenticated
    is_authenticated = True
    get_authenticated_spark(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)

def read_nlu_info(path):
    f = open(os.path.join(path,'nlu_info.txt'), "r")
    nlu_ref = f.readline()
    f.close()
    return nlu_ref


def is_running_in_databricks():
    #Check if the currently running Python Process is running in Databricks or not
    # If any Enviroment Variable name contains 'DATABRICKS' this will return True, otherwise False
    for k in os.environ.keys() :
        if 'DATABRICKS' in k :
            return True
    return False

def load_nlu_pipe_from_hdd(pipe_path):
    pipe = NLUPipeline()
    if nlu.is_running_in_databricks() :
        if pipe_path.startswith('/dbfs/') or pipe_path.startswith('dbfs/'):
            nlu_path = pipe_path
            if pipe_path.startswith('/dbfs/'):
                nlp_path =  pipe_path.replace('/dbfs','')
            else :
                nlp_path =  pipe_path.replace('dbfs','')

        else :
            nlu_path = 'dbfs/' + pipe_path
            if pipe_path.startswith('/') : nlp_path = pipe_path
            else : nlp_path = '/' + pipe_path

        nlu_ref = read_nlu_info(nlu_path)
        if os.path.exists(pipe_path):
            # if os.path.exists(info_path):
            pipe_components = construct_component_from_pipe_identifier(nlu_ref, nlu_ref, 'hdd', path=nlp_path)
            # pipe_components = construct_component_from_pipe_identifier('nlu_ref','nlu_ref','hdd',path=pipe_path)

            for c in pipe_components: pipe.add(c, nlu_ref, pretrained_pipe_component=True)

            return pipe
            # else:
            #     print(f'Could not find nlu_info.json file in  {pipe_path}')
            #     return NluError
        else :
            print(f'Could not find nlu pipe folder in {pipe_path}')
            return NluError


    nlu_ref = read_nlu_info(pipe_path)
    if os.path.exists(pipe_path):
        # if os.path.exists(info_path):
        pipe_components = construct_component_from_pipe_identifier(nlu_ref, 'nlu_ref', 'hdd', path=pipe_path)
        for c in pipe_components: pipe.add(c, nlu_ref, pretrained_pipe_component=True)

        return pipe
        # else:
        #     print(f'Could not find nlu_info.json file in  {pipe_path}')
        #     return NluError
    else :
        print(f'Could not find nlu pipe folder in {pipe_path}')
        return NluError
def enable_verbose():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# TODO IMPLEMENT AUTH PARAMS AND PASS THEM DOWN THE CALL STACK
def load(request ='from_disk', path=None,verbose=False):
    '''
    Load either a prebuild pipeline or a set of components identified by a whitespace seperated list of components
    You must call nlu.auth() BEFORE calling nlu.load() to access licensed models.
    If you did not call nlu.auth() but did call nlu.load() you must RESTART your Python Process and call nlu.auth().
    You cannot authorize once nlu.load() is called because of Spark Context.
    :param verbose:
    :param path: If path is not None, the model/pipe for the NLU reference will be loaded from the path. Useful for offline mode. Currently only loading entire NLU pipelines is supported, but not loading singular pipes
    :param request: A NLU model/pipeline/component reference
    :param version_checks: Wether to check if Pyspark is properly installed and if the Pyspark version is correct for the NLU version. If set to False, these tests will be skipped
    :return: returns a non fitted nlu pipeline object
    '''
    global is_authenticated
    is_authenticated = True
    gc.collect()
    # if version_checks : check_pyspark_install()
    spark = sparknlp.start()
    spark.catalog.clearCache()

    if verbose:
        enable_verbose()
    #
    # try:

    if path != None :
        logger.info(f'Trying to load nlu pipeline from local hard drive, located at {path}')
        pipe = load_nlu_pipe_from_hdd(path)
        return pipe
    components_requested = request.split(' ')
    pipe = NLUPipeline()
    language = parse_language_from_nlu_ref(request)
    pipe.lang=language
    pipe.nlu_ref = request
    for nlu_ref in components_requested:
        nlu_ref.replace(' ', '')
        # component = component.lower()
        if nlu_ref == '': continue
        nlu_component = nlu_ref_to_component(nlu_ref, authenticated=is_authenticated)
        # if we get a list of components, then the NLU reference is a pipeline, we do not need to check order
        if type(nlu_component) == type([]):
            # lists are parsed down to multiple components
            for c in nlu_component: pipe.add(c, nlu_ref, pretrained_pipe_component=True)
        else:
            pipe.add(nlu_component, nlu_ref)
            pipe = PipelineQueryVerifier.check_and_fix_nlu_pipeline(pipe)

    # except:
    #     import sys
    #     e = sys.exc_info()
    #     print(e[0])
    #     print(e[1])
    #
    #     print(
    #         "Something went wrong during loading and fitting the pipe. Check the other prints for more information and also verbose mode. Did you use a correct model reference?")
    #
    #     return NluError()
    active_pipes.append(pipe)
    return pipe

class NluError:
    def __init__(self):
        self.has_trainable_components = False
    @staticmethod
    def predict(text, output_level='error', positions='error', metadata='error', drop_irrelevant_cols=False):
        print('The NLU components could not be properly created. Please check previous print messages and Verbose mode for further info')
    @staticmethod
    def print_info(): print("Sorry something went wrong when building the pipeline. Please check verbose mode and your NLU reference.")



#Discovery
def print_all_languages():
    ''' Print all languages which are available in NLU Spark NLP pointer '''
    discoverer.print_all_languages()
def print_all_nlu_components_for_lang(lang='en', c_type='classifier'):
    '''Print all NLU components available for a language Spark NLP pointer'''
    discoverer.print_all_nlu_components_for_lang(lang, c_type)
def print_all_nlu_components_for_lang(lang='en'):
    '''Print all NLU components available for a language Spark NLP pointer'''
    discoverer.print_all_nlu_components_for_lang(lang)
def print_components(lang='', action=''):
    '''Print every single NLU reference for models and pipeliens and their Spark NLP pointer
    :param lang: Language requirements for the components filterd. See nlu.languages() for supported languages
    :param action: Components that will be filterd.'''
    discoverer.print_components(lang,action)
def print_component_types():
    ''' Prints all unique component types in NLU'''
    discoverer.print_component_types()
def print_all_model_kinds_for_action(action):
    discoverer.print_all_model_kinds_for_action(action)
def print_all_model_kinds_for_action_and_lang(lang, action):
    discoverer.print_all_model_kinds_for_action_and_lang(lang,action)
def print_trainable_components():
    '''Print every trainable Algorithm/Model'''
    discoverer.print_trainable_components()
