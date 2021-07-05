__version__ = '3.1.1'
hard_offline_checks = False
def version(): return __version__
#
# from nlu.component_resolution import parse_language_from_nlu_ref, nlu_ref_to_component, \
#     construct_component_from_pipe_identifier


# if not check_pyspark_install(): raise Exception()
def try_import_pyspark_in_streamlit():
    """Try importing Pyspark or display warn message in streamlit"""
    try :
        import pyspark
        from pyspark.sql import SparkSession
    except:
        print("You need Pyspark installed to run NLU. Run <pip install pyspark==3.0.2>")

        try :
            import streamlit as st
            st.error("You need Pyspark, Sklearn, Pyplot, Pandas, Numpy installed to run this app. Run <pip install pyspark==3.0.2 sklearn pyplot numpy pandas>")
        except:
            return False
        return False
    return True
if not try_import_pyspark_in_streamlit() : raise  ImportError("You ned to install Pyspark")
st_cache_enabled = False
from typing import Optional

import nlu.utils.environment.env_utils as env_utils
import nlu.utils.environment.authentication as auth_utils
if not env_utils.check_python_version(): raise Exception()

import nlu
import logging
from nlu.spellbook import Spellbook
import warnings
from nlu.utils.environment.authentication import *

warnings.filterwarnings("ignore")

logger = logging.getLogger('nlu')
# logger.setLevel(logging.INFO)
logger.setLevel(logging.CRITICAL)
ch = logging.StreamHandler()
ch.setLevel(logging.CRITICAL)

logger.addHandler(ch)

# NLU Healthcare components
from nlu.components.assertion import Asserter
from nlu.components.resolution import Resolver
from nlu.components.relation import Relation
from nlu.components.deidentification import Deidentification

from nlu.components.embeddings.distil_bert.distilbert import DistilBert
from nlu.components.embeddings.roberta.roberta import Roberta
from nlu.components.embeddings.xlm.xlm import XLM



from nlu.components.utils.sentence_embeddings.spark_nlp_sentence_embedding import SparkNLPSentenceEmbeddings

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
from nlu.components.stopwordscleaner import StopWordsCleaner as StopWordsCleaners
from nlu.components.stemmer import Stemmer as Stemmers
from nlu.components.stemmers.stemmer.spark_nlp_stemmer import SparkNLPStemmer
from nlu.components.normalizers.normalizer.spark_nlp_normalizer import SparkNLPNormalizer
from nlu.components.normalizers.document_normalizer.spark_nlp_document_normalizer import SparkNLPDocumentNormalizer

from nlu.components.lemmatizers.lemmatizer.spark_nlp_lemmatizer import SparkNLPLemmatizer
from nlu.components.stopwordscleaners.stopwordcleaner.nlustopwordcleaner import NLUStopWordcleaner
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

from nlu.pipe.pipeline import NLUPipeline
from nlu.pipe.utils.pipe_utils import PipeUtils
from nlu.pipe.pipe_logic import PipelineQueryVerifier
from nlu.pipe.component_resolution import *
global spark, all_components_info, nlu_package_location,authorized
is_authenticated=False
nlu_package_location = nlu.__file__[:-11]

spark_started = False
spark = None
authenticated = False
from nlu.info import AllComponentsInfo
from nlu.discovery import Discoverer
all_components_info = nlu.AllComponentsInfo()
discoverer = nlu.Discoverer()

import  json
import sparknlp
import os

def auth(SPARK_NLP_LICENSE_OR_JSON_PATH='/content/spark_nlp_for_healthcare.json',AWS_ACCESS_KEY_ID='',AWS_SECRET_ACCESS_KEY='',JSL_SECRET='', gpu=False):
    """ Authenticate enviroment for JSL Liscensed models. Installs NLP-Healthcare if not in enviroment detected
    Either provide path to spark_nlp_for_healthcare.json file as first param or manually enter them, SPARK_NLP_LICENSE_OR_JSON_PATH,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET .
    Set gpu=true if you want to enable GPU mode
    """
    if os.path.exists(SPARK_NLP_LICENSE_OR_JSON_PATH):
        with open(SPARK_NLP_LICENSE_OR_JSON_PATH) as json_file:
            j = json.load(json_file)
            auth_utils.get_authenticated_spark(j['SPARK_NLP_LICENSE'],  j['AWS_ACCESS_KEY_ID'],   j['AWS_SECRET_ACCESS_KEY'], j['SECRET'],gpu)
        return nlu
    if AWS_ACCESS_KEY_ID != '':
        auth_utils.get_authenticated_spark(SPARK_NLP_LICENSE_OR_JSON_PATH,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, gpu)
    else : return nlu
    return nlu


def load_nlu_pipe_from_hdd(pipe_path,request)-> NLUPipeline:
    """Either there is a pipeline of models in the path or just one singular model.
    If it is a pipe,  load the pipe and return it.
    If it is a singular model, load it to the correct AnnotatorClass and NLU component and then generate pipeline for it
    """
    pipe = NLUPipeline()
    # if env_utils.is_running_in_databricks() :
    #     if pipe_path.startswith('/dbfs/') or pipe_path.startswith('dbfs/'):
    #         nlu_path = pipe_path
    #         if pipe_path.startswith('/dbfs/'):
    #             nlp_path =  pipe_path.replace('/dbfs','')
    #         else :
    #             nlp_path =  pipe_path.replace('dbfs','')
    #     else :
    #         nlu_path = 'dbfs/' + pipe_path
    #         if pipe_path.startswith('/') : nlp_path = pipe_path
    #         else : nlp_path = '/' + pipe_path
    nlu_ref=request# pipe_path
    if os.path.exists(pipe_path):
        if offline_utils.is_pipe(pipe_path):
            # language, nlp_ref, nlu_ref,path=None, is_licensed=False
            # todo deduct lang and if Licensed or not

            pipe_components = construct_component_from_pipe_identifier('en', nlu_ref, nlu_ref, pipe_path, False)
        elif offline_utils.is_model(pipe_path):
            c = offline_utils.verify_and_create_model(pipe_path)
            c.info.nlu_ref = nlu_ref
            pipe.add(c, nlu_ref, pretrained_pipe_component=True)
            return PipelineQueryVerifier.check_and_fix_nlu_pipeline(pipe)

        else :
            print(f"Could not load model in path {pipe_path}. Make sure the folder contains either a stages subfolder or a metadata subfolder.")
            raise ValueError
        for c in pipe_components: pipe.add(c, nlu_ref, pretrained_pipe_component=True)
        return pipe

    else :
        print(f"Could not load model in path {pipe_path}. Make sure the folder contains either a stages subfolder or a metadata subfolder.")
        raise ValueError


def enable_verbose()-> None:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

def disable_verbose()-> None:
    logger.setLevel(logging.ERROR)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger.addHandler(ch)



def get_open_source_spark_context(gpu):
    if is_env_pyspark_2_3(): return sparknlp.start(spark23=True, gpu=gpu)
    if is_env_pyspark_2_4(): return sparknlp.start(spark24=True, gpu=gpu)
    if is_env_pyspark_3_0()  or is_env_pyspark_3_1(): return sparknlp.start(gpu=gpu)
    print(f"Current Spark version {get_pyspark_version()} not supported!")
    raise ValueError


def enable_streamlit_caching():
    # dynamically monkeypatch aka replace the nlu.load() method with the wrapped st.cache if streamlit_caching
    nlu.st_cache_enabled = True
    nlu.non_caching_load = load
    nlu.load =  wrap_with_st_cache_if_avaiable(nlu.load)

# def disable_streamlit_caching(): # WIP not working
#     if hasattr(nlu, 'non_caching_load') : nlu.load = nlu.non_caching_load
#     else : print("Could not disable caching.")


def wrap_with_st_cache_if_avaiable(f):
    """Wrap function with ST cache method if streamlit is importable"""
    try :
        import streamlit as st
        logger.info("Using streamlit cache for load")
        return st.cache(f, allow_output_mutation=True, show_spinner=False)
    except :
        logger.exception("Could not import streamlit and apply caching")
        print("You need streamlit to run use this method")
        return f


def enable_hard_offline_checks():nlu.hard_offline_checks = True
def disable_hard_offline_checks():nlu.hard_offline_checks = False



def load(request:str ='from_disk', path:Optional[str]=None,verbose:bool=False, gpu:bool=False, streamlit_caching:bool=False)->NLUPipeline :
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
    if streamlit_caching and not nlu.st_cache_enabled :
        enable_streamlit_caching()
        return nlu.load(request, path,verbose, gpu, streamlit_caching)
    global is_authenticated
    is_authenticated = True
    auth(gpu=gpu) # check if secets are in default loc, if yes load them and create licensed context automatically
    spark = get_open_source_spark_context(gpu)
    spark.catalog.clearCache()
    if verbose:enable_verbose()
    else: disable_verbose()


    if path != None :
        logger.info(f'Trying to load nlu pipeline from local hard drive, located at {path}')
        pipe = PipelineQueryVerifier.check_and_fix_nlu_pipeline(load_nlu_pipe_from_hdd(path,request))
        pipe.nlu_ref = request
        return pipe
    components_requested = request.split(' ')
    pipe = NLUPipeline()
    language = parse_language_from_nlu_ref(request)
    pipe.lang=language
    pipe.nlu_ref = request

    try :
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
        pipe.nlu_ref = request
        for c in pipe.components :
            if c.info.license == 'licensed' : pipe.has_licensed_components=True
        return pipe

    except:
        import sys
        if verbose:
            e = sys.exc_info()
            print(e[0])
            print(e[1])

        # 1. Verfiy PYSPARK INSTALLED
        # 2. Verify SPARK-NLP INSTALLED
        # 3. Verify JAVA8 IS DEFAULT
        # FAILURE :
        # IF ST IMPORTABLE, TRY WRITE WARN MESSAGE TO STREAMLIT FROM NLU.LOAD()
        # IF ST NOT IMPORTABLE. JUST PRINT ERR MESSAGE
        # 4. ELSE LINK TO INSTALL NOTES https://nlu.johnsnowlabs.com/docs/en/install  and SLACK https://join.slack.com/t/spark-nlp/shared_invite/zt-lutct9gm-kuUazcyFKhuGY3_0AMkxqA
        raise Exception("Something went wrong during loading and fitting the pipe. Check the other prints for more information and also verbose mode. Did you use a correct model reference?")



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

def get_components(m_type='', include_pipes=False, lang='', licensed=False, get_all=False):
    return discoverer.get_components(m_type, include_pipes, lang, licensed, get_all)
