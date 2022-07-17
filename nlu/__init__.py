__version__ = '4.0.0'

import nlu.utils.environment.env_utils as env_utils

if not env_utils.try_import_pyspark_in_streamlit():
    raise ImportError("You ned to install Pyspark to run nlu. Run pip install pyspark==3.0.1")
if not env_utils.try_import_spark_nlp():
    raise ImportError("You need Spark NLP to run NLU. run pip install spark-nlp")

import json
import sparknlp
import warnings
import nlu.utils.environment.authentication as auth_utils
import nlu.utils.environment.offline_load_utils as offline_utils
from nlu.universe.universes import Licenses
from nlu.utils.environment.authentication import *
from nlu.pipe.pipeline import NLUPipeline
from nlu.pipe.pipe_logic import PipelineQueryVerifier
from nlu.discovery import Discoverer
from nlu.pipe.component_resolution import *


def version(): return __version__


warnings.filterwarnings("ignore")
logger = logging.getLogger('nlu')
logger.setLevel(logging.CRITICAL)
ch = logging.StreamHandler()
ch.setLevel(logging.CRITICAL)
logger.addHandler(ch)
st_cache_enabled = False
nlu_package_location = nlu.__file__[:-11]
discoverer = Discoverer()

slack_link = 'https://join.slack.com/t/spark-nlp/shared_invite/zt-lutct9gm-kuUazcyFKhuGY3_0AMkxqA'
github_issues_link = 'https://github.com/JohnSnowLabs/nlu/issues'




def viz(nlp_pipe: Union[Pipeline, LightPipeline, PipelineModel, List],
        data:str, viz_type: str = '',
        labels_to_viz=None,
        viz_colors={},
        return_html=False,
        ner_col: str = None,
        pos_col: str = None,
        dep_untyped_col: str = None,
        dep_typed_col: str = None,
        resolution_col: str = None,
        relation_col: str = None,
        assertion_col: str = None, ):
    """
Visualize input data with an already configured Spark NLP pipeline,
for  Algorithms of type (Ner,Assertion, Relation, Resolution, Dependency)
using Spark-NLP-Display.
Automatically infers applicable viz type and output columns to use for visualization.
Data may only be a string.


If a pipeline has multiple models candidates that can be used for a viz,
the first Annotator that is vizzable will be used to create viz.
You can specify which type of viz to create with the viz_type parameter

Output columns to use for the viz are automatically deducted from the pipeline, by using the
first pipe that provides the correct output type for a specific viz.
You can specify which columns to use for a viz by using the
corresponding ner_col, pos_col, dep_untyped_col, dep_typed_col, resolution_col, relation_col, assertion_col, parameters.

    :param nlp_pipe: One of  [Pipeline, LightPipeline, PipelineModel, List[Annotator] ]
    :param data:  String to viz
    :param viz_type: Viz type, one of [ner,dep,resolution,relation,assert].
            If none defined, nlu will infer and apply all applicable viz
    :param labels_to_viz: Defines a subset of NER labels to viz i.e. ['PER'] ,
        by default=[] which will display all labels. Applicable only for NER viz

    :param viz_colors: Applicable for [ner, resolution, assert ],
        a dictionary where key = label, value=hex color,
        i.e. viz_colors={'TREATMENT':'#008080', 'problem':'#800080'}
    :param return_html:
    :param ner_col: Specify NER column for NER/Resolution/Assertion viz.
    :param pos_col: Specify POS column for Dependency Tree viz
    :param dep_untyped_col: Specify Untyped Dependency Column for Tree Viz
    :param dep_typed_col: Specify Typed Dependency Column for Tree Viz
    :param resolution_col: Specify Resolution col for resolution viz
    :param relation_col: Specify Relation col for Relation viz
    :param assertion_col: Specify Assertion col for Assertion viz
    :return:
    """
    return to_nlu_pipe(nlp_pipe, True).viz(text_to_viz=data,
                                           viz_type=viz_type,
                                           labels_to_viz=labels_to_viz,
                                           return_html=return_html,
                                           viz_colors=viz_colors,
                                           ner_col=ner_col,
                                           pos_col=pos_col,
                                           dep_untyped_col=dep_untyped_col,
                                           dep_typed_col=dep_typed_col,
                                           resolution_col=resolution_col,
                                           relation_col=relation_col,
                                           assertion_col=assertion_col, )


def autocomplete_pipeline(pipe:Union[Pipeline, LightPipeline, PipelineModel, List], lang='en'):
    """
Auto-Complete a pipeline or single annotator into a runnable pipeline by harnessing NLU's DAG Autocompletion algorithm
and returns it as NLU pipeline.
The standard Spark pipeline is avaiable on the `.vanilla_transformer_pipe` attribute of the returned nlu pipe

Every Annotator and Pipeline of Annotators defines a `DAG` of tasks, with various
dependencies that must be satisfied in `topoligical order`.
NLU enables the completion of an incomplete DAG by finding or creating a path between
the very first input node which is almost always is `DocumentAssembler/MultiDocumentAssembler`
and the very last node(s), which is given by the `topoligical sorting` the iterable annotators parameter.
Paths are created by resolving input features of
 annotators to the corresponding providers with matching storage references.




    :param pipe: Pipeline, list of Annotators or single annotator to complete
    :param lang: Language of the elements in pipe, used to resolve correct language for dependencies of the pipe.
    :return: NLU pipeline with completed pipeline
    """
    # If you dont set lang, you can get storage ref errors!
    if isinstance(pipe, List):
        pipe = to_nlu_pipe(pipe, is_pre_configured=False)
    else:
        pipe = to_nlu_pipe([pipe], is_pre_configured=False)
    pipe = PipelineQueryVerifier.check_and_fix_nlu_pipeline(pipe)
    return pipe


def to_pretty_df(nlp_pipe: Union[Pipeline, LightPipeline, PipelineModel, List], data, positions=False,
                 output_level='', metadata=False, ):
    """
Annotates a Pandas Dataframe/Pandas Series/Numpy Array/Spark DataFrame/Python List strings /Python String
with given Spark NLP pipeline, which is assumed to be complete and runnable.

Annotators are grouped internally by NLU into specific output levels, which will be zipped and exploded together to create
the final output df. Additionally, most  keys from the metadata dictionary in the result annotations will be
expanded into their own columns in the resulting Dataframe.
Some columns are omitted from metadata to reduce total amount of output columns, these can be re-enabled by setting metadata=True


    :param nlp_pipe: Pipeline or List of annotators to use for prediction
    :param data: Data to predict on
    :param output_level: output level, either document/sentence/chunk/token/relation
    :param positions: whether to output indexes that map predictions back to position in origin string
    :param metadata: whether to keep additional metadata in final df or not like
            confidences of every possible class for predictions.
    """
    return to_nlu_pipe(nlp_pipe, True).predict(data, positions=positions,
                                               output_level=output_level,
                                               metadata=metadata)


def to_nlu_pipe(nlp_pipe: Union[Pipeline, LightPipeline, PipelineModel, List], is_pre_configured=True) -> NLUPipeline:
    """
    Convert a pipeline or list which contains sparknlp/sparknlp_jsl annotators
    into NLU pipeline, while maintaining original configuration.
    The pipeline does not need to be pre-fitted.
    :param nlp_pipe: the pipeline to convert, must have iterable attribute of pipe stages.
    :param is_pre_configured: Is the pipeline already configured, i.e. input/output cols are properly matched between anotators? \
            If True, NLU will treat this as a already configured pipeline and will not do any pipeline autocompletion or configs
    :return: nlu pipe wrapping
    """
    pipe = NLUPipeline()
    components = get_nlu_pipe_for_nlp_pipe(nlp_pipe, is_pre_configured)
    for c in components:
        pipe.add(c, is_pre_configured)
        if c.license == Licenses.hc:
            pipe.has_licensed_components = True
    return pipe


def load(request: str = 'from_disk', path: Optional[str] = None, verbose: bool = False, gpu: bool = False,
         streamlit_caching: bool = False,
         m1_chip: bool = False
         ) -> NLUPipeline:
    '''
    Load either a prebuild pipeline or a set of components identified by a whitespace seperated list of components
    You must call nlu.auth() BEFORE calling nlu.load() to access licensed models.
    If you did not call nlu.auth() but did call nlu.load() you must RESTART your Python Process and call nlu.auth().
    You cannot authorize once nlu.load() is called because of Spark Context.
    :param verbose: Whether to output debug prints
    :param gpu: Whether to leverage GPU
    :param streamlit_caching: Whether streamlit caching should be used in Streamlit visualizations. Trade Speed-Up for repeated requests for larger memory usage
    :param path: If path is not None, the model_anno_obj/component_list for the NLU reference will be loaded from the path. Useful for offline mode. Currently only loading entire NLU pipelines is supported, but not loading singular pipes
    :param request: A NLU model_anno_obj/pipeline/component_to_resolve reference. You can request multiple components by separating them with whitespace. I.e. nlu.load('elmo bert albert')
    :return: returns a non fitted nlu pipeline object
    '''
    if streamlit_caching and not nlu.st_cache_enabled:
        enable_streamlit_caching()
        return nlu.load(request, path, verbose, gpu, streamlit_caching)
    # check if secrets are in default loc, if yes load them and create licensed context automatically
    auth(gpu=gpu)
    spark = get_open_source_spark_context(gpu, m1_chip)
    spark.catalog.clearCache()

    if verbose:
        enable_verbose()
    else:

        disable_verbose()
    try:
        if path is not None:
            logger.info(f'Trying to load nlu pipeline from local hard drive, located at {path}')
            pipe = load_nlu_pipe_from_hdd(path, request)
            pipe.nlu_ref = request
            return pipe
    except Exception as err:
        if verbose:
            e = sys.exc_info()
            print(e[0])
            print(e[1])
            print(err)
        raise Exception(
            f"Something while loading the pipe in {path}. Is the path correct? use nlu.load(verbose=True) for more info.")

    # Try to manifest SparkNLP Annotator from nlu_ref
    components_requested = request.split(' ')
    pipe = NLUPipeline()
    language = parse_language_from_nlu_ref(request)
    pipe.lang = language
    pipe.nlu_ref = request
    try:
        for nlu_ref in components_requested:
            # Iterate over each nlu_ref in the request. Multiple nlu_refs can be passed by seperating them via whitesapce
            nlu_ref.replace(' ', '')
            if nlu_ref == '':
                continue
            nlu_component = nlu_ref_to_component(nlu_ref)
            # if we get a list of components, then the NLU reference is a pipeline, we do not need to check order
            if type(nlu_component) == type([]):
                # lists are parsed down to multiple components, result of pipeline request (stack of components)
                for c in nlu_component:
                    pipe.add(c, nlu_ref, pretrained_pipe_component=True)
            else:
                # just a single component_to_resolve requested
                pipe.add(nlu_component, nlu_ref)
    except Exception as err:
        if verbose:
            e = sys.exc_info()
            print(e[0])
            print(e[1])
            print(err)
        raise Exception(
            f"Something went wrong during creating the Spark NLP model_anno_obj for your request =  {request} Did you use a NLU Spell?")
    # Complete Spark NLP Pipeline, which is defined as a DAG given by the starting Annotators
    try:
        pipe = PipelineQueryVerifier.check_and_fix_nlu_pipeline(pipe)
        pipe.nlu_ref = request
        return pipe
    except:
        if verbose:
            e = sys.exc_info()
            print(e[0])
            print(e[1])
        raise Exception(f"Something went wrong during completing the DAG for the Spark NLP Pipeline."
                        f"If this error persists, please contact us in Slack {slack_link} "
                        f"Or open an issue on Github {github_issues_link}")


def auth(HEALTHCARE_LICENSE_OR_JSON_PATH='/content/spark_nlp_for_healthcare.json', AWS_ACCESS_KEY_ID='',
         AWS_SECRET_ACCESS_KEY='', HEALTHCARE_SECRET='', OCR_LICENSE='', OCR_SECRET='', gpu=False):
    """ Authenticate enviroment for JSL Liscensed models.mm
    Installs NLP-Healthcare if not in environment detected
    Either provide path to spark_nlp_for_healthcare.json file as first param or manually enter them,
    HEALTHCARE_LICENSE_OR_JSON_PATH,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,HEALTHCARE_SECRET .
    Set gpu=true if you want to enable GPU mode
    """

    def has_empty_strings(iterable):
        """Check for a given list of strings, whether it has any empty strings or not"""
        return all(x == '' for x in iterable)

    hc_creds = [HEALTHCARE_LICENSE_OR_JSON_PATH, HEALTHCARE_SECRET]
    ocr_creds = [OCR_LICENSE, OCR_SECRET]
    aws_creds = [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]

    if os.path.exists(HEALTHCARE_LICENSE_OR_JSON_PATH):
        # Credentials provided via JSON file
        with open(HEALTHCARE_LICENSE_OR_JSON_PATH) as json_file:
            j = json.load(json_file)
            if 'SPARK_NLP_LICENSE' in j.keys() and 'SPARK_OCR_LICENSE' in j.keys():
                # HC and OCR creds provided
                auth_utils.get_authenticated_spark_HC_and_OCR(j['SPARK_NLP_LICENSE'], j['SECRET'],
                                                              j['SPARK_OCR_LICENSE'], j['SPARK_OCR_SECRET'],
                                                              j['AWS_ACCESS_KEY_ID'], j['AWS_SECRET_ACCESS_KEY'], gpu)

                return nlu

            if 'SPARK_NLP_LICENSE' in j.keys() and 'SPARK_OCR_LICENSE' not in j.keys():
                # HC creds provided but no OCR
                auth_utils.get_authenticated_spark_HC(j['SPARK_NLP_LICENSE'], j['SECRET'], j['AWS_ACCESS_KEY_ID'],
                                                      j['AWS_SECRET_ACCESS_KEY'], gpu)
                return nlu

            if 'SPARK_NLP_LICENSE' not in j.keys() and 'SPARK_OCR_LICENSE' in j.keys():
                # OCR creds provided but no HC
                auth_utils.get_authenticated_spark_OCR(j['SPARK_OCR_LICENSE'], j['SPARK_OCR_SECRET'],
                                                       j['AWS_ACCESS_KEY_ID'], j['AWS_SECRET_ACCESS_KEY'], gpu)
                return nlu

            auth_utils.get_authenticated_spark(gpu)
        return nlu
    else:
        # Credentials provided as parameter
        if not has_empty_strings(hc_creds) and not has_empty_strings(ocr_creds) and not has_empty_strings(aws_creds):
            # HC + OCR credentials provided
            auth_utils.get_authenticated_spark_HC_and_OCR(HEALTHCARE_LICENSE_OR_JSON_PATH, HEALTHCARE_SECRET,
                                                          OCR_LICENSE,
                                                          OCR_SECRET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, gpu)
            return nlu

        elif not has_empty_strings(hc_creds) and has_empty_strings(ocr_creds) and not has_empty_strings(aws_creds):
            # HC creds provided, but no HC
            auth_utils.get_authenticated_spark_HC(HEALTHCARE_LICENSE_OR_JSON_PATH, HEALTHCARE_SECRET, AWS_ACCESS_KEY_ID,
                                                  AWS_SECRET_ACCESS_KEY, gpu)
            return nlu
        elif has_empty_strings(hc_creds) and not has_empty_strings(ocr_creds) and not has_empty_strings(aws_creds):
            # OCR creds provided but no HC
            auth_utils.get_authenticated_spark_OCR(OCR_LICENSE, OCR_SECRET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                                                   gpu)
            return nlu

    return nlu


def load_nlu_pipe_from_hdd(pipe_path, request) -> NLUPipeline:
    """Either there is a pipeline of models in the path or just one singular model_anno_obj.
    If it is a component_list,  load the component_list and return it.
    If it is a singular model_anno_obj, load it to the correct AnnotatorClass and NLU component_to_resolve and then generate pipeline for it
    """
    pipe = NLUPipeline()
    nlu_ref = request  # pipe_path
    if os.path.exists(pipe_path):

        # Ressource in path is a pipeline
        if offline_utils.is_pipe(pipe_path):
            # language, nlp_ref, nlu_ref,path=None, is_licensed=False
            # todo deduct lang and if Licensed or not
            pipe_components = get_trained_component_list_for_nlp_pipe_ref('en', nlu_ref, nlu_ref, pipe_path, False)
        # Resource in path is a single model_anno_obj
        elif offline_utils.is_model(pipe_path):
            c = offline_utils.verify_and_create_model(pipe_path)
            c.nlu_ref = nlu_ref
            pipe.add(c, nlu_ref, pretrained_pipe_component=True)
            return PipelineQueryVerifier.check_and_fix_nlu_pipeline(pipe)
        else:
            print(
                f"Could not load model_anno_obj in path {pipe_path}. Make sure the jsl_folder contains either a stages subfolder or a metadata subfolder.")
            raise ValueError
        for c in pipe_components: pipe.add(c, nlu_ref, pretrained_pipe_component=True)
        return pipe

    else:
        print(
            f"Could not load model_anno_obj in path {pipe_path}. Make sure the jsl_folder contains either a stages subfolder or a metadata subfolder.")
        raise ValueError


def get_open_source_spark_context(gpu, m1_chip):
    if env_utils.is_env_pyspark_3_x():
        if m1_chip:
            return sparknlp.start(gpu=gpu, m1=True)
        else:
            return sparknlp.start(gpu=gpu)
    raise ValueError(f"Failure starting Spark Context! Current Spark version {get_pyspark_version()} not supported! "
                     f"Please install any of Pyspark 3.X versions.")


def enable_verbose() -> None:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


def disable_verbose() -> None:
    logger.setLevel(logging.ERROR)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger.addHandler(ch)


def enable_streamlit_caching():
    # dynamically monkeypatch aka replace the nlu.load() method with the wrapped st.cache if streamlit_caching
    nlu.st_cache_enabled = True
    nlu.non_caching_load = load
    nlu.load = wrap_with_st_cache_if_available_and_set_layout_to_wide(nlu.load)


# def disable_streamlit_caching(): # WIP not working
#     if hasattr(nlu, 'non_caching_load') : nlu.load = nlu.non_caching_load
#     else : print("Could not disable caching.")

# TODO EXPORT
def wrap_with_st_cache_if_available_and_set_layout_to_wide(f):
    """Wrap function with ST cache method if streamlit is importable"""
    try:
        import streamlit as st
        st.set_page_config(layout='wide')
        logger.info("Using streamlit cache for load")
        return st.cache(f, allow_output_mutation=True, show_spinner=False)
    except:
        logger.exception("Could not import streamlit and apply caching")
        print("You need streamlit to run use this method")
        return f


# Discovery
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
    discoverer.print_components(lang, action)


def print_component_types():
    ''' Prints all unique component_to_resolve types in NLU'''
    discoverer.print_component_types()


def print_all_model_kinds_for_action(action):
    discoverer.print_all_model_kinds_for_action(action)


def print_all_model_kinds_for_action_and_lang(lang, action):
    discoverer.print_all_model_kinds_for_action_and_lang(lang, action)


def print_trainable_components():
    '''Print every trainable Algorithm/Model'''
    discoverer.print_trainable_components()


def get_components(m_type='', include_pipes=False, lang='', licensed=False, get_all=False):
    return discoverer.get_components(m_type, include_pipes, lang, licensed, get_all)

# https://forms.gle/VZeJRLBDM6m9fhF68
