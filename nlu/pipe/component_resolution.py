'''
Contains methods used to resolve a NLU reference to a NLU component_to_resolve.
Handler for getting default components, etc.
'''
from typing import Dict, List, Union, Optional, Callable

from pyspark.ml import PipelineModel, Pipeline
from sparknlp.pretrained import PretrainedPipeline, LightPipeline

from nlu.pipe.nlu_component import NluComponent
from nlu.pipe.utils.component_utils import ComponentUtils
from nlu.pipe.utils.pipe_utils import PipeUtils
from nlu.pipe.utils.resolution.storage_ref_resolution_utils import *
from nlu.spellbook import Spellbook
from nlu.universe.atoms import LicenseType
from nlu.universe.component_universes import ComponentUniverse, anno_class_to_empty_component
from nlu.universe.feature_resolutions import FeatureResolutions
from nlu.universe.feature_universes import NLP_HC_FEATURES, OCR_FEATURES
from nlu.universe.universes import Licenses, license_to_bucket, ModelBuckets

logger = logging.getLogger('nlu')


def init_component(component):
    # Init partial constructor
    if isinstance(component, Callable):
        component = component()
    return component

def is_produced_by_multi_output_component(missing_feature_type: Union[NLP_FEATURES, OCR_FEATURES, NLP_HC_FEATURES]):
    """For these components we resolve to None,
    because they are already beeign satisfied by another component that outputs multiple features
    including the ones coverd here """
    return missing_feature_type == NLP_FEATURES.DOCUMENT_QUESTION_CONTEXT



def resolve_feature(missing_feature_type: Union[NLP_FEATURES, OCR_FEATURES, NLP_HC_FEATURES], language='en',
                    is_licensed=False,
                    is_trainable_pipe=False) -> NluComponent:
    '''
    This function returns a default component_to_resolve for a missing component_to_resolve type
     and core part to the pipeline feature resolution.
    It is used to auto complete pipelines, which are missing required components.
    :param missing_feature_type: String which is either just the component_to_resolve
    type or componenttype@spark_nlp_reference which stems from a models storageref and refers to some pretrained
    embeddings or model_anno_obj
    :return: a NLU component_to_resolve which is a either the default if there is no '@' in the @param
    missing_component_type or a default component_to_resolve for that particular type
    '''
    logger.info(f'Getting default for missing_feature_type={missing_feature_type}')
    if is_produced_by_multi_output_component(missing_feature_type) :
        return None
    if '@' not in missing_feature_type:
        # Resolve feature which has no storage ref or if storage ref is irrelevant at this point
        if is_licensed and is_trainable_pipe and missing_feature_type in FeatureResolutions.default_HC_train_resolutions.keys():
            feature_resolution = FeatureResolutions.default_HC_resolutions[missing_feature_type]
            license_type = Licenses.hc
            model_bucket = ModelBuckets.hc
        elif is_licensed and missing_feature_type in FeatureResolutions.default_HC_resolutions.keys():
            feature_resolution = FeatureResolutions.default_HC_resolutions[missing_feature_type]
            license_type = Licenses.hc
            model_bucket = ModelBuckets.hc
        elif is_licensed and missing_feature_type in FeatureResolutions.default_OCR_resolutions.keys():
            feature_resolution = FeatureResolutions.default_OCR_resolutions[missing_feature_type]
            license_type = Licenses.ocr
            # model_bucket = 'clinical/models' # no bucket based models supported
            model_bucket = ModelBuckets.ocr
        elif missing_feature_type in FeatureResolutions.default_OS_resolutions.keys():
            feature_resolution = FeatureResolutions.default_OS_resolutions[missing_feature_type]
            license_type = Licenses.open_source
            model_bucket = ModelBuckets.open_source
        else:
            raise ValueError(f"Could not resolve feature={missing_feature_type}")
        nlu_component = init_component(feature_resolution.nlu_component)  # Call the partial and init the nlu component

        # Either call get_pretrained(nlp_ref, lang,bucket) or get_default_model() to instantiate Annotator object
        if feature_resolution.get_pretrained:
            return nlu_component.set_metadata(
                nlu_component.get_pretrained_model(feature_resolution.nlp_ref, feature_resolution.language,
                                                   model_bucket),
                feature_resolution.nlu_ref, feature_resolution.nlp_ref, language, False, license_type)
        else:
            return nlu_component.set_metadata(nlu_component.get_default_model(),
                                              feature_resolution.nlu_ref, feature_resolution.nlp_ref, language, False,
                                              license_type)

    else:
        # if there is an @ in the name, we must get some specific
        # pretrained model_anno_obj from the sparknlp reference that should follow after the @
        missing_feature_type, storage_ref = missing_feature_type.split('@')

        if storage_ref == '':
            # Storage ref empty for trainable resolution.
            # Use default embed defined in feature resolution
            if is_licensed and is_trainable_pipe and missing_feature_type in FeatureResolutions.default_HC_train_resolutions.keys():
                feature_resolution = FeatureResolutions.default_HC_resolutions[missing_feature_type]
                license_type = Licenses.hc
                model_bucket = ModelBuckets.hc
            elif missing_feature_type in FeatureResolutions.default_OS_resolutions.keys():
                feature_resolution = FeatureResolutions.default_OS_resolutions[missing_feature_type]
                license_type = Licenses.open_source
                model_bucket = ModelBuckets.open_source
            else:
                raise ValueError(
                    f"Could not resolve empty storage ref with default feature for missing feature = {missing_feature_type}")
            nlu_component = init_component(
                feature_resolution.nlu_component)  # Call the partial and init the nlu component
            return nlu_component.set_metadata(
                nlu_component.get_pretrained_model(feature_resolution.nlp_ref, feature_resolution.language,
                                                   model_bucket),
                feature_resolution.nlu_ref, feature_resolution.nlp_ref, language, False, license_type)

        # Actually resolve storage ref
        nlu_ref, nlp_ref, is_licensed, language = resolve_storage_ref(language, storage_ref, missing_feature_type)
        license_type = Licenses.hc if is_licensed else Licenses.open_source
        nlu_component = get_trained_component_for_nlp_model_ref(language, nlu_ref, nlp_ref, license_type)
        return nlu_component


def nlu_ref_to_component(nlu_ref, detect_lang=False, authenticated=False) -> NluComponent:
    '''
    This method implements the main namespace for all component_to_resolve names. It parses the input request and passes
    the data to a resolver method which searches the namespace for a Component for the input request
    It returns a list of NLU.component_to_resolve objects or just one NLU.component_to_resolve
    object alone if just one component_to_resolve was specified.
    It maps a correctly namespaced name to a corresponding component_to_resolve for pipeline
    If no lang is provided, default language eng is assumed.
    General format  <lang>.<class>.<dataset>.<embeddings>
    For embedding format : <lang>.<class>.<variant>
    This method will parse <language>.<NLU_action>
        Additional data about dataset and variant will be resolved by corrosponding action classes

    If train prefix is part of the nlu_ref ,the trainable namespace will e searched

    if 'translate_to' or 'marian' is inside the nlu_ref, 'xx' will be prefixed to the ref and set as lang if it is not already
    Since all translate models are xx lang
    :param nlu_ref: User request (should be a NLU reference)
    :param detect_lang: Whether to automatically  detect language
    :return: Pipeline or component_to_resolve for the NLU reference.
    '''

    infos = nlu_ref.split('.')
    if len(infos) == 0:
        raise ValueError(f"EXCEPTION: Could not create a component_to_resolve for nlu reference={nlu_ref}", )

    if 'train' in infos:
        if nlu_ref in Spellbook.trainable_models.keys():
            if nlu_ref not in Spellbook.trainable_models:
                s = "\n"
                raise ValueError(f'Could not find trainable model_anno_obj for nlu_ref={nlu_ref}.'
                                 f'Supported values = {s.join(nlu.Spellbook.trainable_models.keys())}')
            return get_trainable_component_for_nlu_ref(nlu_ref)
    lang, nlu_ref, nlp_ref, license_type, is_pipe, model_params = nlu_ref_to_nlp_metadata(nlu_ref)

    if is_pipe:
        resolved_component = get_trained_component_list_for_nlp_pipe_ref(lang, nlp_ref, nlu_ref,
                                                                         license_type=license_type)
    else:
        resolved_component = get_trained_component_for_nlp_model_ref(lang, nlu_ref, nlp_ref, license_type, model_params)

    if resolved_component is None:
        raise ValueError(f"EXCEPTION: Could not create a component_to_resolve for nlu reference={nlu_ref}", )
    return resolved_component


def get_trainable_component_for_nlu_ref(nlu_ref) -> NluComponent:
    if nlu_ref in Spellbook.traianble_nlu_ref_to_jsl_anno_id:
        anno_id = Spellbook.traianble_nlu_ref_to_jsl_anno_id[nlu_ref]
    else:
        raise ValueError(f'Could not find trainable Model for nlu_spell ={nlu_ref}')
    if anno_id in ComponentUniverse.components:
        component = ComponentUniverse.components[anno_id]()
        return component.set_metadata(component.get_trainable_model(), nlu_ref, '', 'xx', False )
    else:
        raise ValueError(f'Could not find trainable Model for anno_id ={anno_id}')


def get_trained_component_list_for_nlp_pipe_ref(language, nlp_ref, nlu_ref, path=None,
                                                license_type: LicenseType = Licenses.open_source,
                                                ) -> List[NluComponent]:
    """
    creates a list of components from a Spark NLP Pipeline reference
    1. download pipeline
    2. unpack pipeline to annotators and create list of nlu components
    3. return list of nlu components
    :param license_type: Type of license for the component
    :param nlu_ref: Nlu ref that points to this pipe
    :param language: language of the pipeline
    :param nlp_ref: Reference to a spark nlp pretrained pipeline
    :param path: Load component_list from HDD
    :return: Each element of the Spark NLP pipeline wrapped as a NLU component_to_resolve inside a list
    """
    logger.info(f'Building pretrained pipe for nlu_ref={nlu_ref} nlp_ref={nlp_ref}')
    if 'language' in nlp_ref:
        # special edge case for lang detectors
        language = 'xx'
    if path is None:
        if license_type != Licenses.open_source:
            pipe = PretrainedPipeline(nlp_ref, lang=language, remote_loc='clinical/models')
        else:
            pipe = PretrainedPipeline(nlp_ref, lang=language)
        iterable_stages = pipe.light_model.pipeline_model.stages
    else:
        pipe = LightPipeline(PipelineModel.load(path=path))
        iterable_stages = pipe.pipeline_model.stages
    constructed_components = get_component_list_for_iterable_stages(iterable_stages, language, nlp_ref, nlu_ref,
                                                                    license_type)
    return ComponentUtils.set_storage_ref_attribute_of_embedding_converters(
        PipeUtils.set_column_values_on_components_from_pretrained_pipe(constructed_components, nlp_ref, language, path))


def get_nlu_pipe_for_nlp_pipe(pipe: Union[Pipeline, LightPipeline, PipelineModel, List], is_pre_configured=True):
    """Get a list of NLU components wrapping each Annotator in pipe.
    Pipe should be of class Pipeline, LightPipeline, or PipelineModel
    :param pipe: for which to extract list of nlu components which embellish each annotator
    :return: list of nlu components, one per annotator in pipe
    """

    if isinstance(pipe, List):
        pipe = get_component_list_for_iterable_stages(pipe, is_pre_configured=is_pre_configured)
    elif isinstance(pipe, Pipeline):
        pipe = get_component_list_for_iterable_stages(pipe.getStages(), is_pre_configured=is_pre_configured)
    elif isinstance(pipe, LightPipeline):
        pipe = get_component_list_for_iterable_stages(pipe.pipeline_model.stages, is_pre_configured=is_pre_configured)
    elif isinstance(pipe, PipelineModel):
        pipe = get_component_list_for_iterable_stages(pipe.stages, is_pre_configured=is_pre_configured)
    elif isinstance(pipe, PretrainedPipeline):
        pipe = get_component_list_for_iterable_stages(pipe.model.stages, is_pre_configured=is_pre_configured)
    else:
        raise ValueError(
            f'Invalid Pipe-Like class {type(pipe)} supported types: Pipeline,LightPipeline,PipelineModel,List')
    if is_pre_configured:
        return set_cols_on_nlu_components(pipe)
    else:
        return pipe


def set_cols_on_nlu_components(iterable_components):
    for c in iterable_components:
        c.spark_input_column_names = c.model.getInputCols() if hasattr(c.model, 'getInputCols') else [
            c.model.getInputCol()]
        if hasattr(c.model, 'getOutputCol'):
            c.spark_output_column_names = [c.model.getOutputCol()]
        elif hasattr(c.model, 'getOutputCols'):
            c.spark_output_column_names = [c.model.getOutputCols()]

    return iterable_components


def get_component_list_for_iterable_stages(iterable_stages, language=None, nlp_ref=None, nlu_ref=None,
                                           is_pre_configured=True
                                           ):
    constructed_components = []
    for jsl_anno_object in iterable_stages:
        anno_class_name = type(jsl_anno_object).__name__
        logger.info(f"Building NLU component for class_name = {anno_class_name} ")
        component = anno_class_to_empty_component(anno_class_name)

        component.set_metadata(jsl_anno_object, nlu_ref, nlp_ref, language, is_pre_configured)
        constructed_components.append(component)
        if None in constructed_components or len(constructed_components) == 0:
            raise Exception(f"Failure inferring type anno_class={anno_class_name} ")
    return constructed_components


def get_trained_component_for_nlp_model_ref(lang: str, nlu_ref: Optional[str] = '', nlp_ref: str = '',
                                            license_type: LicenseType = Licenses.open_source,
                                            model_configs: Optional[Dict[str, any]] = None) -> NluComponent:
    anno_class = Spellbook.nlp_ref_to_anno_class[nlp_ref]
    component = anno_class_to_empty_component(anno_class)
    model_bucket = license_to_bucket(license_type)
    try:
        if component.get_pretrained_model:
            component = component.set_metadata(
                component.get_pretrained_model(nlp_ref, lang, model_bucket),
                nlu_ref, nlp_ref, lang, False, license_type)
        else:
            component = component.set_metadata(component.get_default_model(),
                                               nlu_ref, nlp_ref, lang, False, license_type)
        if model_configs:
            for method_name, parameter in model_configs.items():
                # Dynamically call method from provided name and value, to set parameters like T5 task
                code = f'component.model.{method_name}({parameter})'
                eval(code)
    except Exception as e:
        raise ValueError(f'Failure making component, nlp_ref={nlp_ref}, nlu_ref={nlu_ref}, lang={lang}, \n err={e}')

    return component
