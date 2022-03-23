'''
Contains methods used to resolve a NLU reference to a NLU component_to_resolve.
Handler for getting default components, etc.
'''
from pyspark.ml import PipelineModel

import nlu.utils.environment.authentication as auth_utils
from nlu.pipe.nlu_component import NluComponent
from nlu.universe.annotator_class_universe import AnnoClassRef
from nlu.pipe.utils.pipe_utils import PipeUtils
from nlu.pipe.utils.component_utils import ComponentUtils
from typing import Union, List
from nlu.pipe.utils.resolution.storage_ref_resolution_utils import *
from nlu.spellbook import Spellbook
from sparknlp.pretrained import PretrainedPipeline, LightPipeline

from nlu.universe.feature_node_ids import NLP_NODE_IDS, NLP_HC_NODE_IDS
from nlu.universe.universes import Licenses
from nlu.universe.component_universes import ComponentMap
from nlu.universe.feature_resolutions import FeatureResolutions
import nlu

logger = logging.getLogger('nlu')


def resolve_feature(missing_feature_type, language='en', is_licensed=False,
                    is_trainable_pipe=False) -> NluComponent:
    '''
    This function returns a default component_to_resolve for a missing component_to_resolve type and core part to the pipeline feature resolution.
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
        nlu_component = copy.copy(ComponentMap.os_components[jsl_anno_id])
        # We write storage ref to nlu_component, for the case of accumulated chunk and sentence embeddings.
        # Anno Class has no storage ref in these cases, but it is still an embedding provider
        return nlu_component.set_metadata(nlu_component.get_pretrained_model(nlp_ref, language, model_bucket),
                                          nlu_ref,
                                          nlp_ref, language,
                                          False, license_type, storage_ref)

        # if 'pos' in missing_feature_type or 'ner' in missing_feature_type:
        #     return construct_component_from_identifier(language=language, component_type='classifier',
        #                                                nlp_ref=storage_ref)


def nlu_ref_to_component(nlu_reference, detect_lang=False, authenticated=False,
                         is_recursive_call=False) -> NluComponent:
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
    :param nlu_reference: User request (should be a NLU reference)
    :param detect_lang: Wether to automatically  detect language
    :return: Pipeline or component_to_resolve for the NLU reference.
    '''

    infos = nlu_reference.split('.')
    if len(infos) < 0:
        logger.exception(f"EXCEPTION: Could not create a component_to_resolve for nlu reference={nlu_reference}", )
        return nlu.NluError()
    language = ''
    component_type = ''
    dataset = ''
    component_embeddings = ''
    trainable = False
    resolved_component = None
    # 1. Check if either a default cmponent or one specific pretrained component_to_resolve or component_list  or alias of them is is requested without more sepcificatin about lang,dataset or embeding.
    # I.e. 'explain_ml' , 'explain; 'albert_xlarge_uncased' or ' tokenize'  or 'sentiment' s requested. in this case, every possible annotator must be checked.
    # format <class> or <nlu identifier>
    # type parsing via checking parts in nlu action <>

    if len(infos) == 0:
        logger.exception("Split on query is 0.")
    # Query of format <class>, no embeds,lang or dataset specified
    elif 'train' in infos:
        trainable = True
        component_type = infos[1]
    elif 'translate_to' in nlu_reference and not 't5' in nlu_reference:
        # Special case for translate_to, will be prefixed with 'xx' if not already prefixed with xx because there are only xx.<ISO>.translate_to references
        language = 'xx'
        if nlu_reference[0:3] != 'xx.':
            nlu_reference = 'xx.' + nlu_reference
        logger.info(f'Setting lang as xx for nlu_ref={nlu_reference}')
        component_type = 'translate_to'
    elif len(infos) == 1:
        # if we only have 1 split result, it must a a NLU action reference or an alias
        logger.info('Setting default lang to english')
        language = 'en'
        if infos[0] in nlu.all_components_info.all_components or nlu.all_components_info.all_nlu_actions:
            component_type = infos[0]
    #  check if it is any query of style #<lang>.<class>.<dataset>.<embeddings>
    elif infos[0] in nlu.all_components_info.all_languages:
        language = infos[0]
        component_type = infos[1]
        logger.info(f"Got request for trained model {component_type}")
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

    # After parsing base metadata from the NLU ref, we construct the Spark NLP Annotator
    logger.info(
        f'For input nlu_ref {nlu_reference} detected : \n {language}:   , component_to_resolve type: {component_type} , component_to_resolve dataset:{dataset}, component_to_resolve embeddings {component_embeddings} ')
    resolved_component = resolve_component_from_parsed_query_data(language, component_type, dataset,
                                                                  component_embeddings, nlu_reference, trainable,
                                                                  authenticated=authenticated,
                                                                  is_recursive_call=is_recursive_call)
    if resolved_component is None:
        logger.exception("EXCEPTION: Could not create a component_to_resolve for nlu reference=%s", nlu_reference)
        return nlu.NluError()
    return resolved_component


def resolve_component_from_parsed_query_data(lang, component_type, dataset, component_embeddings, nlu_ref,
                                             trainable=False, path=None, authenticated=False,
                                             is_recursive_call=False) -> Union[
    NluComponent, List[NluComponent]]:  # NLUPipeline
    '''
    Searches the NLU name spaces for a matching NLU reference. From that NLU reference, a SparkNLP reference will be aquired which resolved to a SparkNLP pretrained model or pipeline
    :param nlu_ref: Full request which was passed to nlu.load()
    :param lang: parsed language, may never be  '' and should be default 'en'
    :param component_type: parsed component_to_resolve type. may never be ''
    :param dataset: parsed dataset, can be ''
    :param component_embeddings: parsed embeddigns used for the component_to_resolve, can be ''
    :return: returns the nlu.Component class that corrosponds to this component_to_resolve. If it is a pretrained pipeline, it will return a list of components(?)
    '''
    component_kind = ''  # either model or component_list or auto_pipe
    nlp_ref = ''
    logger.info('Searching local Namespaces for SparkNLP reference.. ')
    resolved = False
    is_licensed = False
    # 0. check trainable references
    if trainable == True:
        if nlu_ref in Spellbook.trainable_models.keys():
            component_kind = 'trainable_model'
            nlp_ref = Spellbook.trainable_models[nlu_ref]
            logger.info(f'Found Spark NLP reference in trainable models naqmespace = {nlp_ref}')
            resolved = True
            trainable = True
    # 1. check if pipeline references for resolution
    if resolved == False and lang in Spellbook.pretrained_pipe_references.keys():
        if nlu_ref in Spellbook.pretrained_pipe_references[lang].keys():
            component_kind = 'component_list'
            nlp_ref = Spellbook.pretrained_pipe_references[lang][nlu_ref]
            logger.info(f'Found Spark NLP reference in pretrained pipelines namespace = {nlp_ref}')
            resolved = True

    # 2. check if model references for resolution
    if resolved == False and lang in Spellbook.pretrained_models_references.keys():
        if nlu_ref in Spellbook.pretrained_models_references[lang].keys():
            component_kind = 'model'
            nlp_ref = Spellbook.pretrained_models_references[lang][nlu_ref]
            logger.info(f'Found Spark NLP reference in pretrained models namespace = {nlp_ref}')
            resolved = True

    # 3. check if alias/default references for resolution
    if resolved == False and nlu_ref in Spellbook.component_alias_references.keys():
        sparknlp_data = Spellbook.component_alias_references[nlu_ref]
        component_kind = sparknlp_data[1]
        nlp_ref = sparknlp_data[0]
        logger.info('Found Spark NLP reference in language free aliases namespace')
        resolved = True
        lang = 'en'
        if len(sparknlp_data) > 2:
            dataset = sparknlp_data[2]
        if len(sparknlp_data) > 3:
            # special case overwrite for T5
            nlu_ref = sparknlp_data[3]
        # Check if alias is referring to multi lingual model and set lang accordingly

    # 4. Check Healthcare Pipe Namespace
    if resolved == False and lang in Spellbook.pretrained_healthcare_pipe_references.keys():
        if nlu_ref in Spellbook.pretrained_healthcare_pipe_references[lang].keys():
            component_kind = 'component_list'
            nlp_ref = Spellbook.pretrained_healthcare_pipe_references[lang][nlu_ref]
            logger.info(f'Found Spark NLP reference in pretrained healthcare component_list namespace = {nlp_ref}')
            resolved = True
            is_licensed = True

    # 5. Check Healthcare Model Namespace
    if resolved == False and lang in Spellbook.pretrained_healthcare_model_references.keys():
        if nlu_ref in Spellbook.pretrained_healthcare_model_references[lang].keys():
            component_kind = 'model'
            nlp_ref = Spellbook.pretrained_healthcare_model_references[lang][nlu_ref]
            logger.info(f'Found Spark NLP reference in pretrained healthcare model namespace = {nlp_ref}')
            resolved = True
            is_licensed = True

    # 6. Check Healthcare Aliases Namespace
    if resolved == False and nlu_ref in Spellbook.healthcare_component_alias_references.keys():
        sparknlp_data = Spellbook.healthcare_component_alias_references[nlu_ref]
        component_kind = sparknlp_data[1]
        nlp_ref = sparknlp_data[0]
        logger.info('Found Spark NLP reference in language free healthcare aliases namespace')
        resolved = True
        is_licensed = True
        # if len(sparknlp_data) > 2 :
        #     dataset=sparknlp_data[2]
        # if len(sparknlp_data) > 3 :
        #     # special case overwrite for T5
        #     nlu_ref=sparknlp_data[3]

    # 7. Check OCR NameSpace
    if resolved == False and nlu_ref in Spellbook.ocr_model_references.keys():
        component_kind = 'model'
        nlp_ref = Spellbook.ocr_model_references[nlu_ref]
        logger.info(f'Found Spark OCR reference in OCR model namespace = {nlp_ref}')
        resolved = True
        is_licensed = True

    # 7. If reference is none of the Namespaces, it must be a component_to_resolve like tokenizer or YAKE or Chunker etc....
    # If it is not, then it does not exist and will be caught later
    if not resolved:
        resolved = True
        component_kind = 'component_to_resolve'
        logger.info(
            'Could not find reference in NLU namespace. Assuming it is a component_to_resolve that is an ragmatic NLP annotator with NO model to download behind it.')

    if not auth_utils.is_authorized_environment() and is_licensed:
        print(f"The nlu_ref=[{nlu_ref}] is pointing to a licensed Spark NLP Annotator or Model [{nlp_ref}]. \n"
              f"Your environment does not seem to be Authorized!\n"
              f"Please RESTART your Python environment and run nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)  \n"
              f"with your corrosponding credentials. If you have no credentials you can get a trial version with credentials here https://www.johnsnowlabs.com/spark-nlp-try-free/ \n"
              f"Or contact us at contact@jonsnowlabs.com\n"
              f"NLU will ignore this error and continue running, but you will encounter errors most likely. ")

    # 8. If requested component_to_resolve is NER and it could not yet be resolved to a default NER model,
    # We check if the lang is coverd by multi_ner_base or multi_ner_EXTREME
    if component_type == 'ner' and nlp_ref == '':
        if lang in nlu.all_components_info.all_multi_lang_base_ner_languages:
            lang = 'xx'
            nlp_ref = 'ner_wikiner_glove_840B_300'
            nlu_ref = 'xx.ner.wikiner_glove_840B_300'
        if lang in nlu.all_components_info.all_multi_lang_xtreme_ner_languages:
            lang = 'xx'
            nlp_ref = 'ner_xtreme_glove_840B_300'
            nlu_ref = 'xx.ner.xtreme_glove_840B_300'

    if nlp_ref == '' and not is_recursive_call and not trainable and 'en.' not in nlu_ref and lang in ['', 'en']:
        # Search again but with en. prefixed, enables all refs to work withouth en prefix
        return nlu_ref_to_component('en.' + nlu_ref, is_recursive_call=True)

    # Convert references into NLU Component object which embelishes NLP annotators
    if component_kind == 'component_list':
        constructed_components = construct_component_from_pipe_identifier(lang, nlp_ref, nlu_ref,
                                                                          is_licensed=is_licensed)
        logger.info(
            f'Inferred Spark reference nlp_ref={nlp_ref} and nlu_ref={nlu_ref}  to NLP Annotator Class {constructed_components}')
        if constructed_components is None:
            raise ValueError(
                f'EXCEPTION : Could not create NLU component_to_resolve for nlp_ref={nlp_ref} and nlu_ref={nlu_ref}  to NLP Annotator Classes {constructed_components}')
        else:
            return constructed_components
    elif component_kind in ['model', 'component_to_resolve']:
        constructed_component = construct_component_from_identifier(lang, component_type, dataset,
                                                                    component_embeddings, nlu_ref,
                                                                    nlp_ref, is_licensed=is_licensed)

        logger.info(
            f'Inferred Spark reference nlp_ref={nlp_ref} and nlu_ref={nlu_ref}  to NLP Annotator Class {constructed_component}')

        if constructed_component is None:
            raise ValueError(f'EXCEPTION : Could not create NLU component_to_resolve for nlp_ref={nlp_ref} and nlu_ref={nlu_ref}')
        else:
            return constructed_component
    elif component_kind == 'trainable_model':
        constructed_component = construct_trainable_component_from_identifier(nlu_ref, nlp_ref)
        if constructed_component is None:
            raise ValueError(f'EXCEPTION : Could not create NLU component_to_resolve for nlp_ref={nlp_ref} and nlu_ref={nlu_ref}')
        else:
            # constructed_component.is_untrained = True
            return constructed_component
    else:
        raise ValueError(f'EXCEPTION : Could not create NLU component_to_resolve for nlp_ref={nlp_ref} and nlu_ref={nlu_ref}')


def construct_trainable_component_from_identifier(nlu_ref, nlp_ref) -> NluComponent:
    '''
    This method returns a Spark NLP annotator Approach class embelished by a NLU component_to_resolve
    :param nlu_ref: nlu ref to the trainable model
    :param nlp_ref: nlp ref to the trainable model
    :return: trainable model as a NLU component_to_resolve
    '''
    logger.info(f'Creating trainable NLU component_to_resolve for nlu_ref = {nlu_ref} and nlp_ref = {nlp_ref}')

    if nlu_ref in Spellbook.traianble_nlu_ref_to_jsl_anno_id.keys():
        anno_id = Spellbook.traianble_nlu_ref_to_jsl_anno_id[nlu_ref]
    else:
        raise ValueError(f'Could not find trainable Model for nlu_spell ={nlu_ref}')

    try:
        if anno_id in ComponentMap.os_components.keys():
            nlu_component = ComponentMap.os_components[anno_id]
            return nlu_component.set_metadata(nlu_component.get_trainable_model(), nlu_ref, nlp_ref, 'xx', False,
                                              Licenses.open_source)
        elif anno_id in ComponentMap.hc_components.keys():
            nlu_component = ComponentMap.hc_components[anno_id]
            return nlu_component.set_metadata(nlu_component.get_trainable_model(), nlu_ref, nlp_ref, 'xx', False,
                                              Licenses.hc)

        else:
            raise ValueError(f'Could not find trainable Model for nlu_spell ={nlu_ref}')

    except:  # if reference is not in namespace and not a component_to_resolve it will cause a unrecoverable crash
        ValueError(
            f'EXCEPTION: Could not create trainable NLU component_to_resolve for nlu_ref = {nlu_ref} and nlp_ref = {nlp_ref}')


def construct_component_from_pipe_identifier(language, nlp_ref, nlu_ref, path=None,
                                             is_licensed=False):  # -> NLUPipeline
    """
    creates a list of components from a Spark NLP Pipeline reference
    1. download pipeline
    2. unpack pipeline to annotators and create list of nlu components
    3. return list of nlu components
    :param is_licensed: Weather pipe is licensed or not
    :param nlu_ref: Nlu ref that points to this pipe
    :param language: language of the pipeline
    :param nlp_ref: Reference to a spark nlp pretrained pipeline
    :param path: Load component_list from HDD
    :return: Each element of the Spark NLP pipeline wrapped as a NLU component_to_resolve inside a list
    """
    if 'language' in nlp_ref:
        # special edge case for lang detectors
        language = 'xx'
    if path is None:
        if is_licensed:
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
            nlu_component = ComponentMap.os_components[jsl_anno_id]
            nlu_component.set_metadata(jsl_anno_object, nlu_ref, nlp_ref, language, True, Licenses.open_source)
            constructed_components.append(nlu_component)
        elif anno_class_name in hc_annos.keys():
            # Licensed HC
            jsl_anno_id = hc_annos[anno_class_name]
            nlu_component = ComponentMap.hc_components[jsl_anno_id]
            nlu_component.set_metadata(jsl_anno_object, nlu_ref, nlp_ref, language, True, Licenses.hc)
            constructed_components.append(nlu_component)
        elif anno_class_name in ocr_annos:
            # Licensed OCR
            jsl_anno_id = ocr_annos[anno_class_name]
            nlu_component = ComponentMap.ocr_components[jsl_anno_id]
            nlu_component.set_metadata(jsl_anno_object, nlu_ref, nlp_ref, language, True, Licenses.ocr)
            constructed_components.append(nlu_component)
        else:
            raise ValueError(f'Could not find matching nlu component_to_resolve for annotator class = {anno_class_name}')
        if None in constructed_components or len(constructed_components) == 0:
            raise Exception(f"Failure inferring type anno_class={anno_class_name} ")
    return ComponentUtils.set_storage_ref_attribute_of_embedding_converters(
        PipeUtils.set_column_values_on_components_from_pretrained_pipe(constructed_components, nlp_ref, language, path))


def construct_component_from_identifier(language, component_type='', dataset='', component_embeddings='', nlu_ref='',
                                        nlp_ref='', is_licensed=False, anno_class_name = None) -> NluComponent:
    '''
    Creates a NLU component_to_resolve from a pretrained SparkNLP model reference or Class reference. First step to get the Root of the NLP DAG
    Class references will return default pretrained models
    :param language: Language of the sparknlp model reference
    :param component_type: Class which will be used to instantiate the model
    :param dataset: Dataset that the model was trained on
    :param component_embeddings: Embedded that the models was traiend on (if any)
    :param nlu_ref: Full user request
    :param nlp_ref: Full Spark NLP reference
    :return: Returns a NLU component_to_resolve which embelished the Spark NLP pretrained model and class for that model
    '''
    if not anno_class_name:
        # anno_class_name != None for models loaded from Disk, otherwise we need to infer it
        anno_class_name = Spellbook.nlp_ref_to_anno_class[nlp_ref]
    os_annos = AnnoClassRef.get_os_pyclass_2_anno_id_dict()
    hc_annos = AnnoClassRef.get_hc_pyclass_2_anno_id_dict()
    ocr_annos = AnnoClassRef.get_ocr_pyclass_2_anno_id_dict()
    logger.info(
        f'Creating component_to_resolve, sparknlp_ref={nlp_ref}, nlu_ref={nlu_ref},language={language} ')
    model_bucket = 'clinical/models' if is_licensed else None
    try:
        if anno_class_name in os_annos.keys():
            # Open Source
            jsl_anno_id = os_annos[anno_class_name]
            nlu_component = ComponentMap.os_components[jsl_anno_id]
            if nlu_component.get_pretrained_model:
                return nlu_component.set_metadata(nlu_component.get_pretrained_model(nlp_ref, language, model_bucket),
                                                  nlu_ref, nlp_ref,
                                                  language,
                                                  False, Licenses.open_source)
            else:
                return nlu_component.set_metadata(nlu_component.get_default_model(),
                                                  nlu_ref, nlp_ref,
                                                  language,
                                                  False, Licenses.open_source)

        elif anno_class_name in hc_annos.keys():
            # Licensed HC
            jsl_anno_id = hc_annos[anno_class_name]
            nlu_component = ComponentMap.hc_components[jsl_anno_id]
            if nlu_component.get_pretrained_model:
                return nlu_component.set_metadata(
                    nlu_component.get_pretrained_model(nlp_ref, language, 'clinical/models'),
                    nlu_ref,
                    nlp_ref, language,
                    False, Licenses.hc)
            else:
                return nlu_component.set_metadata(nlu_component.get_default_model(),
                                                  nlu_ref,
                                                  nlp_ref, language,
                                                  False, Licenses.hc)

        elif anno_class_name in ocr_annos.keys():
            # Licensed OCR (WIP)
            jsl_anno_id = ocr_annos[anno_class_name]
            nlu_component = ComponentMap.ocr_components[jsl_anno_id]
            if nlu_component.get_pretrained_model:

                return nlu_component.set_metadata(nlu_component.get_pretrained_model(nlp_ref, language, ), nlu_ref,
                                                  nlp_ref, language,
                                                  False, Licenses.ocr)
            else:
                # Model with no pretrained weights
                return nlu_component.set_metadata(nlu_component.get_default_model(), nlu_ref,
                                                  nlp_ref, language,
                                                  False, Licenses.ocr)

        else:
            raise ValueError(f'Could not find matching nlu component_to_resolve for annotator class = {anno_class_name}')
    except Exception as e:
        raise ValueError(f'Failure build component_to_resolve, nlp_ref={nlp_ref}, nlu_ref={nlu_ref}, lang ={language}, err ={e}')
