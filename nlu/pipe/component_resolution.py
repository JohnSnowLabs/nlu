'''
Contains methods used to resolve a NLU reference to a NLU component_to_resolve.
Handler for getting default components, etc.
'''
from typing import Dict

from pyspark.ml import PipelineModel
from sparknlp.pretrained import PretrainedPipeline, LightPipeline

from nlu.pipe.nlu_component import NluComponent
from nlu.pipe.utils.component_utils import ComponentUtils
from nlu.pipe.utils.pipe_utils import PipeUtils
from nlu.pipe.utils.resolution.storage_ref_resolution_utils import *
from nlu.spellbook import Spellbook
from nlu.universe.annotator_class_universe import AnnoClassRef
from nlu.universe.atoms import LicenseType
from nlu.universe.component_universes import ComponentUniverse
from nlu.universe.feature_resolutions import FeatureResolutions
from nlu.universe.universes import Licenses

logger = logging.getLogger('nlu')


def resolve_feature(missing_feature_type, language='en', is_licensed=False,
                    is_trainable_pipe=False) -> NluComponent:
    '''
    This function returns a default component_to_resolve for a missing component_to_resolve type
     and core part to the pipeline feature resolution.
    It is used to auto complete pipelines, which are missing required components.
    :param missing_feature_type: String which is either just the component_to_resolve
    type or componenttype@spark_nlp_reference which stems from a models storageref and refers to some pretrained
    embeddings or model
    :return: a NLU component_to_resolve which is a either the default if there is no '@' in the @param
    missing_component_type or a default component_to_resolve for that particular type
    '''
    logger.info(f'Getting default for missing_feature_type={missing_feature_type}')
    model_bucket = 'clinical/models' if is_licensed else None
    if '@' not in missing_feature_type:
        # Resolve feature which has no storage ref or if storage ref is irrelevant at this point
        if is_licensed and is_trainable_pipe and missing_feature_type in FeatureResolutions.default_HC_train_resolutions.keys():
            feature_resolution = FeatureResolutions.default_HC_resolutions[missing_feature_type]
            license_type = Licenses.hc
            model_bucket = 'clinical/models'
        elif is_licensed and missing_feature_type in FeatureResolutions.default_HC_resolutions.keys():
            feature_resolution = FeatureResolutions.default_HC_resolutions[missing_feature_type]
            license_type = Licenses.hc
            model_bucket = 'clinical/models'
        elif is_licensed and missing_feature_type in FeatureResolutions.default_OCR_resolutions.keys():
            feature_resolution = FeatureResolutions.default_OCR_resolutions[missing_feature_type]
            license_type = Licenses.ocr
            # model_bucket = 'clinical/models' # no bucket based models supported
        elif missing_feature_type in FeatureResolutions.default_OS_resolutions.keys():
            feature_resolution = FeatureResolutions.default_OS_resolutions[missing_feature_type]
            license_type = Licenses.open_source
            model_bucket = None
        else:
            raise ValueError(f"Could not resolve feature={missing_feature_type}")
        nlu_component = feature_resolution.nlu_component  # Substitution to keep lines short
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
        # if there is an @ in the name, we must get some specific pretrained model from the sparknlp reference that should follow after the @
        missing_feature_type, storage_ref = missing_feature_type.split('@')

        if storage_ref == '':
            # Storage ref empty for trainable resolution.
            # Use default embed defined in feature resolution
            if is_licensed and is_trainable_pipe and missing_feature_type in FeatureResolutions.default_HC_train_resolutions.keys():
                feature_resolution = FeatureResolutions.default_HC_resolutions[missing_feature_type]
                license_type = Licenses.hc
                model_bucket = 'clinical/models'
            elif missing_feature_type in FeatureResolutions.default_OS_resolutions.keys():
                feature_resolution = FeatureResolutions.default_OS_resolutions[missing_feature_type]
                license_type = Licenses.open_source
                model_bucket = None
            else:
                raise ValueError(
                    f"Could not resolve empty storage ref with default feature for missing feature = {missing_feature_type}")
            nlu_component = feature_resolution.nlu_component  # Substitution to keep lines short
            return nlu_component.set_metadata(
                nlu_component.get_pretrained_model(feature_resolution.nlp_ref, feature_resolution.language,
                                                   model_bucket), feature_resolution.nlu_ref,
                feature_resolution.nlp_ref, language, False, license_type)

        nlu_ref, nlp_ref, is_licensed, language = resolve_storage_ref(language, storage_ref, missing_feature_type)
        anno_class_name = Spellbook.nlp_ref_to_anno_class[nlp_ref]
        # All storage ref providers are defined in open source
        os_annos = AnnoClassRef.get_os_pyclass_2_anno_id_dict()
        license_type = Licenses.hc if is_licensed else Licenses.open_source
        model_bucket = 'clinical/models' if is_licensed else None
        jsl_anno_id = os_annos[anno_class_name]
        import copy
        nlu_component = copy.copy(ComponentUniverse.os_components[jsl_anno_id])
        # We write storage ref to nlu_component, for the case of accumulated chunk and sentence embeddings.
        # Anno Class has no storage ref in these cases, but it is still an embedding provider
        return nlu_component.set_metadata(nlu_component.get_pretrained_model(nlp_ref, language, model_bucket),
                                          nlu_ref,
                                          nlp_ref, language,
                                          False, license_type, storage_ref)


def nlu_ref_to_component(nlu_ref, detect_lang=False, authenticated=False) -> NluComponent:
    '''
    This method implements the main namespace for all component_to_resolve names. It parses the input request and passes the data to a resolver method which searches the namespace for a Component for the input request
    It returns a list of NLU.component_to_resolve objects or just one NLU.component_to_resolve object alone if just one component_to_resolve was specified.
    It maps a correctly namespaced name to a corresponding component_to_resolve for pipeline
    If no lang is provided, default language eng is assumed.
    General format  <lang>.<class>.<dataset>.<embeddings>
    For embedding format : <lang>.<class>.<variant>
    This method will parse <language>.<NLU_action>
        Additional data about dataset and variant will be resolved by corrosponding action classes

    If train prefix is part of the nlu_ref ,the trainable namespace will e searched

    if 'translate_to' or 'marian' is inside of the nlu_ref, 'xx' will be prefixed to the ref and set as lang if it is not already
    Since all translate models are xx lang
    :param nlu_ref: User request (should be a NLU reference)
    :param detect_lang: Wether to automatically  detect language
    :return: Pipeline or component_to_resolve for the NLU reference.
    '''

    infos = nlu_ref.split('.')
    if len(infos) == 0:
        raise ValueError(f"EXCEPTION: Could not create a component_to_resolve for nlu reference={nlu_ref}", )

    if 'train' in infos:
        if nlu_ref in Spellbook.trainable_models.keys():
            if nlu_ref not in Spellbook.trainable_models:
                s = "\n"
                raise ValueError(f'Could not find trainable model for nlu_ref={nlu_ref}.'
                                 f'Supported values = {s.join(nlu.Spellbook.trainable_models.keys())}')
            # TODO ,nlp ref for traianble?
            return construct_trainable_component_from_identifier(nlu_ref)
    lang, nlu_ref, nlp_ref, license_type, is_pipe,model_params = nlu_ref_to_nlp_metadata(nlu_ref)

    if is_pipe:
        resolved_component = construct_component_from_pipe_identifier(lang, nlp_ref, nlu_ref, license_type=license_type)
    else:
        resolved_component = construct_component_from_identifier(lang, nlu_ref, nlp_ref, license_type,model_params)

    if resolved_component is None:
        raise ValueError(f"EXCEPTION: Could not create a component_to_resolve for nlu reference={nlu_ref}", )
    return resolved_component


def construct_trainable_component_from_identifier(nlu_ref, nlp_ref='') -> NluComponent:
    '''
    This method returns a Spark NLP annotator Approach class embelished by a NLU component_to_resolve
    :param nlu_ref: nlu ref to the trainable model
    :param nlp_ref: nlp ref to the trainable model
    :return: trainable model as a NLU component_to_resolve
    '''
    logger.info(f'Creating trainable NLU component_to_resolve for nlu_ref = {nlu_ref} ')

    if nlu_ref in Spellbook.traianble_nlu_ref_to_jsl_anno_id.keys():
        anno_id = Spellbook.traianble_nlu_ref_to_jsl_anno_id[nlu_ref]
    else:
        raise ValueError(f'Could not find trainable Model for nlu_spell ={nlu_ref}')

    try:
        if anno_id in ComponentUniverse.os_components.keys():
            nlu_component = ComponentUniverse.os_components[anno_id]
            return nlu_component.set_metadata(nlu_component.get_trainable_model(), nlu_ref, nlp_ref, 'xx', False,
                                              Licenses.open_source)
        elif anno_id in ComponentUniverse.hc_components.keys():
            nlu_component = ComponentUniverse.hc_components[anno_id]
            return nlu_component.set_metadata(nlu_component.get_trainable_model(), nlu_ref, nlp_ref, 'xx', False,
                                              Licenses.hc)

        else:
            raise ValueError(f'Could not find trainable Model for nlu_spell ={nlu_ref}')

    except Exception:  # if reference is not in namespace and not a component_to_resolve it will cause a unrecoverable crash
        ValueError(
            f'EXCEPTION: Could not create trainable NLU component_to_resolve for nlu_ref = {nlu_ref} and nlp_ref = {nlp_ref}')


def construct_component_from_pipe_identifier(language, nlp_ref, nlu_ref, path=None,
                                             license_type: LicenseType = Licenses.open_source):  # -> NLUPipeline
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
    logger.info(f'Building pretrained pipe nlu_ref={nlu_ref} nlp_ref={nlp_ref}')
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
    constructed_components = []
    os_annos = AnnoClassRef.get_os_pyclass_2_anno_id_dict()
    hc_annos = AnnoClassRef.get_hc_pyclass_2_anno_id_dict()
    ocr_annos = AnnoClassRef.get_ocr_pyclass_2_anno_id_dict()
    for jsl_anno_object in iterable_stages:
        anno_class_name = type(jsl_anno_object).__name__
        logger.info(
            f"Extracting model from Spark NLP pipeline: obj= {jsl_anno_object} class_name = {anno_class_name} and creating Component")
        if anno_class_name in os_annos.keys():
            jsl_anno_id = os_annos[anno_class_name]
            nlu_component = ComponentUniverse.os_components[jsl_anno_id]
            nlu_component.set_metadata(jsl_anno_object, nlu_ref, nlp_ref, language, True, Licenses.open_source)
            constructed_components.append(nlu_component)
        elif anno_class_name in hc_annos.keys():
            # Licensed HC
            jsl_anno_id = hc_annos[anno_class_name]
            nlu_component = ComponentUniverse.hc_components[jsl_anno_id]
            nlu_component.set_metadata(jsl_anno_object, nlu_ref, nlp_ref, language, True, Licenses.hc)
            constructed_components.append(nlu_component)
        elif anno_class_name in ocr_annos:
            # Licensed OCR
            jsl_anno_id = ocr_annos[anno_class_name]
            nlu_component = ComponentUniverse.ocr_components[jsl_anno_id]
            nlu_component.set_metadata(jsl_anno_object, nlu_ref, nlp_ref, language, True, Licenses.ocr)
            constructed_components.append(nlu_component)
        else:
            raise ValueError(
                f'Could not find matching nlu component_to_resolve for annotator class = {anno_class_name}')
        if None in constructed_components or len(constructed_components) == 0:
            raise Exception(f"Failure inferring type anno_class={anno_class_name} ")
    return ComponentUtils.set_storage_ref_attribute_of_embedding_converters(
        PipeUtils.set_column_values_on_components_from_pretrained_pipe(constructed_components, nlp_ref, language, path))


def construct_component_from_identifier(language: str, nlu_ref: str = '', nlp_ref: str = '',
                                        license_type: LicenseType = Licenses.open_source,
                                        model_configs: Dict[str, any] = {}) -> NluComponent:
    '''
    Creates a NLU component_to_resolve from a pretrained SparkNLP model reference or Class reference. First step to get the Root of the NLP DAG
    Class references will return default pretrained models
    :param language: Language of the sparknlp model reference
    :param nlu_ref: Full user request
    :param nlp_ref: Full Spark NLP reference
    :param license_type: Type of license for the component
    :return: Returns a new NLU component
    '''
    anno_class_name = Spellbook.nlp_ref_to_anno_class[nlp_ref]
    os_annos = AnnoClassRef.get_os_pyclass_2_anno_id_dict()
    hc_annos = AnnoClassRef.get_hc_pyclass_2_anno_id_dict()
    ocr_annos = AnnoClassRef.get_ocr_pyclass_2_anno_id_dict()
    logger.info(
        f'Creating component_to_resolve, sparknlp_ref={nlp_ref}, nlu_ref={nlu_ref},language={language} ')
    model_bucket = 'clinical/models' if license_type != Licenses.open_source else None
    try:
        if anno_class_name in os_annos.keys():
            # Open Source
            jsl_anno_id = os_annos[anno_class_name]
            nlu_component = ComponentUniverse.os_components[jsl_anno_id]
            if nlu_component.get_pretrained_model:
                component = nlu_component.set_metadata(
                    nlu_component.get_pretrained_model(nlp_ref, language, model_bucket),
                    nlu_ref, nlp_ref,
                    language,
                    False, Licenses.open_source)
            else:
                component = nlu_component.set_metadata(nlu_component.get_default_model(),
                                                       nlu_ref, nlp_ref,
                                                       language,
                                                       False, Licenses.open_source)

        elif anno_class_name in hc_annos.keys():
            # Licensed HC
            jsl_anno_id = hc_annos[anno_class_name]
            nlu_component = ComponentUniverse.hc_components[jsl_anno_id]
            if nlu_component.get_pretrained_model:
                component = nlu_component.set_metadata(
                    nlu_component.get_pretrained_model(nlp_ref, language, 'clinical/models'),
                    nlu_ref,
                    nlp_ref, language,
                    False, Licenses.hc)
            else:
                component = nlu_component.set_metadata(nlu_component.get_default_model(),
                                                       nlu_ref,
                                                       nlp_ref, language,
                                                       False, Licenses.hc)

        elif anno_class_name in ocr_annos.keys():
            # Licensed OCR (WIP)
            jsl_anno_id = ocr_annos[anno_class_name]
            nlu_component = ComponentUniverse.ocr_components[jsl_anno_id]
            if nlu_component.get_pretrained_model:

                component = nlu_component.set_metadata(nlu_component.get_pretrained_model(nlp_ref, language, ), nlu_ref,
                                                       nlp_ref, language,
                                                       False, Licenses.ocr)
            else:
                # Model with no pretrained weights
                component = nlu_component.set_metadata(nlu_component.get_default_model(), nlu_ref,
                                                       nlp_ref, language,
                                                       False, Licenses.ocr)

        else:
            raise ValueError(f'Failure making component for annotator class = {anno_class_name}')
        if model_configs:
            for method_name, parameter in model_configs.items():
                # Dynamically call method from provided name and value, to set parameters like T5 task
                code = f'component.model.{method_name}({parameter})'
                eval(code)
        return component
    except Exception as e:
        raise ValueError(f'Failure making component, nlp_ref={nlp_ref}, nlu_ref={nlu_ref}, lang={language}, \n err={e}')
