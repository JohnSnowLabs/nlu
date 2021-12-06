'''
Contains methods used to resolve a NLU reference to a NLU component.
Handler for getting default components, etcc.
'''

import nlu.utils.environment.authentication as auth_utils
import nlu.utils.environment.offline_load_utils as offline_utils
from pyspark.ml import PipelineModel
from sparknlp.annotator import *
import nlu
from nlu import logger, Util, Embeddings, Classifier, Spellbook, ClassifierDl, NLUSentenceDetector, NGram, Seq2Seq, \
    SpellChecker, Matcher, Nlu_Tokenizer
from nlu.components import embeddings_chunker
from nlu.components.labeled_dependency_parser import LabeledDependencyParser as LabledDepParser
from nlu.components.unlabeled_dependency_parser import UnlabeledDependencyParser as UnlabledDepParser
from nlu.pipe.utils.pipe_utils import PipeUtils
from nlu.pipe.utils.component_utils import ComponentUtils
from sparknlp.base import *
from typing import Union
from nlu.pipe.utils.resolution.storage_ref_resolution_utils import *
from nlu.pipe.pipe_component import NLUComponent


def get_default_component_of_type(missing_component_type, language='en', is_licensed=False,
                                  is_trainable_pipe=False) -> NLUComponent:
    '''
    This function returns a default component for a missing component type and core part to the pipeline resolution/
    It is used to auto complete pipelines, which are missng required components.
    These represents defaults for many applications and should be set wisely.
    :param missing_component_type: String which is either just the component type or componenttype@spark_nlp_reference which stems from a models storageref and refers to some pretrained embeddings or model
    :return: a NLU component which is a either the default if there is no '@' in the @param missing_component_type or a default component for that particualar type
    '''
    logger.info(f'Getting default for missing_component_type={missing_component_type}')

    if not '@' in missing_component_type:
        # get default models if there is no @ in the model name included and ignore storage ref handling
        if missing_component_type == 'document': return Util('document_assembler', nlu_ref='document')
        if missing_component_type == 'sentence': return Util('deep_sentence_detector', nlu_ref='sentence')
        if missing_component_type == 'sentence_embeddings': return Embeddings('use', nlu_ref='embed_sentence.use')
        if 'token' in missing_component_type: return Nlu_Tokenizer("default_tokenizer",
                                                                   nlu_ref='tokenize',
                                                                   language=language)
        if missing_component_type == 'word_embeddings': return Embeddings(annotator_class='glove',
                                                                          nlu_ref='embed.glove')
        if missing_component_type == 'pos':   return Classifier(nlu_ref='pos')
        if missing_component_type == 'ner':   return Classifier(nlu_ref='ner') if not is_licensed else Classifier(
            nlu_ref='med_ner', nlp_ref='jsl_ner_wip_clinical')
        if missing_component_type == 'ner_converter':   return Util('ner_converter', nlu_ref='entity')
        if missing_component_type == 'chunk': return nlu.chunker.Chunker(nlu_ref='chunker')
        if missing_component_type == 'ngram': return nlu.chunker.Chunker(nlu_ref='ngram')
        if missing_component_type == 'chunk_embeddings': return embeddings_chunker.EmbeddingsChunker(
            nlu_ref='chunk_embeddings')
        if missing_component_type == 'unlabeled_dependency': return UnlabledDepParser(nlu_ref='dep.untyped')
        if missing_component_type == 'labled_dependency': return LabledDepParser('dep', nlu_ref='dep.typed')
        if missing_component_type == 'date': return nlu.Matcher('date', nlu_ref='match.date')
        if missing_component_type == 'ner_chunk': return Util('ner_converter', nlu_ref='entitiy')
        if missing_component_type == 'entities' and is_licensed and is_trainable_pipe: return Util('doc2chunk')
        if missing_component_type == 'entities' and is_licensed: return Util('ner_to_chunk_converter_licensed')
        if missing_component_type == 'entities': return Util('ner_converter')
        if missing_component_type == 'feature_vector': return Util('feature_assembler')
        if missing_component_type == 'chunk_2_doc': return Util('chunk_2_doc')

    else:
        """ These models are fetched becuase they required a storage ref, so we ad a storage ref attribute to the component info"""
        # if there is an @ in the name, we must get some specific pretrained model from the sparknlp reference that should follow after the @
        missing_component_type, storage_ref = missing_component_type.split('@')

        if 'embed' in missing_component_type:
            # Get storage ref and metadata
            nlu_ref, nlp_ref, is_licensed, language = resolve_storage_ref(language, storage_ref, missing_component_type)
            if 'chunk_embeddings' in missing_component_type:
                return set_storage_ref_and_resolution_on_component_info(
                    embeddings_chunker.EmbeddingsChunker(nlu_ref=nlu_ref, nlp_ref=nlp_ref), storage_ref)
            else:
                return set_storage_ref_and_resolution_on_component_info(
                    construct_component_from_identifier(language=language, component_type='embed', nlu_ref=nlu_ref,
                                                        nlp_ref=nlp_ref, is_licensed=is_licensed), storage_ref, )

        if 'pos' in missing_component_type or 'ner' in missing_component_type:
            return construct_component_from_identifier(language=language, component_type='classifier',
                                                       nlp_ref=storage_ref)
        logger.exception(f"Could not resolve default component type for missing type={missing_component_type}")


def nlu_ref_to_component(nlu_reference, detect_lang=False, authenticated=False,
                         is_recursive_call=False) -> NLUComponent:
    '''
    This method implements the main namespace for all component names. It parses the input request and passes the data to a resolver method which searches the namespace for a Component for the input request
    It returns a list of NLU.component objects or just one NLU.component object alone if just one component was specified.
    It maps a correctly namespaced name to a corresponding component for pipeline
    If no lang is provided, default language eng is assumed.
    General format  <lang>.<class>.<dataset>.<embeddings>
    For embedding format : <lang>.<class>.<variant>
    This method will parse <language>.<NLU_action>
        Additional data about dataset and variant will be resolved by corrosponding action classes

    If train prefix is part of the nlu_reference ,the trainable namespace will e searched

    if 'translate_to' or 'marian' is inside of the nlu_reference, 'xx' will be prefixed to the ref and set as lang if it is not already
    Since all translate models are xx lang
    :param nlu_reference: User request (should be a NLU reference)
    :param detect_lang: Wether to automatically  detect language
    :return: Pipeline or component for the NLU reference.
    '''

    infos = nlu_reference.split('.')

    if len(infos) < 0:
        logger.exception(f"EXCEPTION: Could not create a component for nlu reference={nlu_reference}", )
        return nlu.NluError()
    language = ''
    component_type = ''
    dataset = ''
    component_embeddings = ''
    trainable = False
    resolved_component = None
    # 1. Check if either a default cmponent or one specific pretrained component or pipe  or alias of them is is requested without more sepcificatin about lang,dataset or embeding.
    # I.e. 'explain_ml' , 'explain; 'albert_xlarge_uncased' or ' tokenize'  or 'sentiment' s requested. in this case, every possible annotator must be checked.
    # format <class> or <nlu identifier>
    # type parsing via checking parts in nlu action <>

    if len(infos) == 0:
        logger.exception("Split  on query is 0.")
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

    logger.info(
        f'For input nlu_ref {nlu_reference} detected : \n {language}:   , component type: {component_type} , component dataset:{dataset}, component embeddings {component_embeddings} ')
    resolved_component = resolve_component_from_parsed_query_data(language, component_type, dataset,
                                                                  component_embeddings, nlu_reference, trainable,
                                                                  authenticated=authenticated,
                                                                  is_recursive_call=is_recursive_call)
    if resolved_component is None:
        logger.exception("EXCEPTION: Could not create a component for nlu reference=%s", nlu_reference)
        return nlu.NluError()
    return resolved_component


def resolve_component_from_parsed_query_data(lang, component_type, dataset, component_embeddings, nlu_ref,
                                             trainable=False, path=None, authenticated=False,
                                             is_recursive_call=False) -> Union[NLUComponent]:  # NLUPipeline
    '''
    Searches the NLU name spaces for a matching NLU reference. From that NLU reference, a SparkNLP reference will be aquired which resolved to a SparkNLP pretrained model or pipeline
    :param nlu_ref: Full request which was passed to nlu.load()
    :param lang: parsed language, may never be  '' and should be default 'en'
    :param component_type: parsed component type. may never be ''
    :param dataset: parsed dataset, can be ''
    :param component_embeddings: parsed embeddigns used for the component, can be ''
    :return: returns the nlu.Component class that corrosponds to this component. If it is a pretrained pipeline, it will return a list of components(?)
    '''
    component_kind = ''  # either model or pipe or auto_pipe
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
            component_kind = 'pipe'
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
        if len(sparknlp_data) > 2:
            dataset = sparknlp_data[2]
        if len(sparknlp_data) > 3:
            # special case overwrite for T5
            nlu_ref = sparknlp_data[3]

    # 4. Check Healthcare Pipe Namespace
    if resolved == False and lang in Spellbook.pretrained_healthcare_pipe_references.keys():
        if nlu_ref in Spellbook.pretrained_healthcare_pipe_references[lang].keys():
            component_kind = 'pipe'
            nlp_ref = Spellbook.pretrained_healthcare_pipe_references[lang][nlu_ref]
            logger.info(f'Found Spark NLP reference in pretrained healthcare pipe namespace = {nlp_ref}')
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

    # 7. If reference is none of the Namespaces, it must be a component like tokenizer or YAKE or Chunker etc....
    # If it is not, then it does not exist and will be caught later
    if not resolved:
        resolved = True
        component_kind = 'component'
        logger.info(
            'Could not find reference in NLU namespace. Assuming it is a component that is an ragmatic NLP annotator with NO model to download behind it.')

    if not auth_utils.is_authorized_enviroment() and is_licensed:
        print(f"The nlu_ref=[{nlu_ref}] is pointing to a licensed Spark NLP Annotator or Model [{nlp_ref}]. \n"
              f"Your environment does not seem to be Authorized!\n"
              f"Please RESTART your Python environment and run nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)  \n"
              f"with your corrosponding credentials. If you have no credentials you can get a trial version with credentials here https://www.johnsnowlabs.com/spark-nlp-try-free/ \n"
              f"Or contact us at contact@jonsnowlabs.com\n"
              f"NLU will ignore this error and continue running, but you will encounter errors most likely. ")

    # 8. If requested component is NER and it could not yet be resolved to a default NER model,
    #   We check if the lang is coverd by multi_ner_base or multi_ner_EXTREME
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
    if component_kind == 'pipe':
        constructed_components = construct_component_from_pipe_identifier(lang, nlp_ref, nlu_ref,
                                                                          is_licensed=is_licensed)
        logger.info(
            f'Inferred Spark reference nlp_ref={nlp_ref} and nlu_ref={nlu_ref}  to NLP Annotator Class {constructed_components}')
        if constructed_components is None:
            raise ValueError(
                f'EXCEPTION : Could not create NLU component for nlp_ref={nlp_ref} and nlu_ref={nlu_ref}  to NLP Annotator Classes {constructed_components}')
        else:
            return constructed_components
    elif component_kind in ['model', 'component']:
        constructed_component = construct_component_from_identifier(lang, component_type, dataset,
                                                                    component_embeddings, nlu_ref,
                                                                    nlp_ref, is_licensed=is_licensed)

        logger.info(
            f'Inferred Spark reference nlp_ref={nlp_ref} and nlu_ref={nlu_ref}  to NLP Annotator Class {constructed_component}')

        if constructed_component is None:
            raise ValueError(f'EXCEPTION : Could not create NLU component for nlp_ref={nlp_ref} and nlu_ref={nlu_ref}')
        else:
            return constructed_component
    elif component_kind == 'trainable_model':
        constructed_component = construct_trainable_component_from_identifier(nlu_ref, nlp_ref, authenticated)
        if constructed_component is None:
            raise ValueError(f'EXCEPTION : Could not create NLU component for nlp_ref={nlp_ref} and nlu_ref={nlu_ref}')
        else:
            constructed_component.info.is_untrained = True
            return constructed_component
    else:
        raise ValueError(f'EXCEPTION : Could not create NLU component for nlp_ref={nlp_ref} and nlu_ref={nlu_ref}')


def construct_trainable_component_from_identifier(nlu_ref, nlp_ref, authenticated=False) -> NLUComponent:
    '''
    This method returns a Spark NLP annotator Approach class embelished by a NLU component
    :param nlu_ref: nlu ref to the trainable model
    :param nlp_ref: nlp ref to the trainable model
    :return: trainable model as a NLU component
    '''

    logger.info(f'Creating trainable NLU component for nlu_ref = {nlu_ref} and nlp_ref = {nlp_ref}')
    try:
        if nlu_ref in ['train.deep_sentence_detector', 'train.sentence_detector']:
            # no label col but trainable?
            return nlu.NLUSentenceDetector(annotator_class='deep_sentence_detector', trainable='True',
                                           nlu_ref=nlu_ref, )
        if nlu_ref in ['train.context_spell', 'train.spell']:
            pass
        if nlu_ref in ['train.symmetric_spell']:
            pass
        if nlu_ref in ['train.norvig_spell']:
            pass
        if nlu_ref in ['train.unlabeled_dependency_parser']:
            pass
        if nlu_ref in ['train.labeled_dependency_parser']:
            pass
        if nlu_ref in ['train.classifier_dl', 'train.classifier']:
            return nlu.Classifier(annotator_class='classifier_dl', trainable=True, nlu_ref=nlu_ref, )
        if nlu_ref in ['train.ner', 'train.named_entity_recognizer_dl']:
            return nlu.Classifier(annotator_class='ner', trainable=True, nlu_ref=nlu_ref, )
        if nlu_ref in ['train.sentiment_dl', 'train.sentiment']:
            return nlu.Classifier(annotator_class='sentiment_dl', trainable=True, nlu_ref=nlu_ref, )
        if nlu_ref in ['train.vivekn_sentiment']:
            pass
        if nlu_ref in ['train.pos']:
            return nlu.Classifier(annotator_class='pos', trainable=True, nlu_ref=nlu_ref, )
        if nlu_ref in ['train.multi_classifier']:
            return nlu.Classifier(annotator_class='multi_classifier', trainable=True, nlu_ref=nlu_ref, )
        if nlu_ref in ['train.word_seg', 'train.word_segmenter']:
            return nlu.Nlu_Tokenizer(annotator_class='word_segmenter', trainable=True, nlu_ref=nlu_ref, )

        if nlu_ref in ['train.generic_classifier']:
            if not authenticated: print(
                "WARNING! You are trying to train a Licensed Model and your environment does not seem to be Authenticated.\n"
                "Please restart your runtime and run nlu.auth() and for more details see https://nlu.johnsnowlabs.com/docs/en/examples_hc#authorize-access-to-licensed-features-and-install-healthcare-dependencies  ")

            return nlu.Classifier(annotator_class='generic_classifier', trainable=True, nlu_ref=nlu_ref,
                                  is_licensed=True)

        if nlu_ref in ['train.resolve_chunks']:
            if not authenticated: print(
                "WARNING! You are trying to train a Licensed Model and your environment does not seem to be Authenticated.\n"
                "Please restart your runtime and run nlu.auth() and for more details see https://nlu.johnsnowlabs.com/docs/en/examples_hc#authorize-access-to-licensed-features-and-install-healthcare-dependencies  ")

            return nlu.Resolver(annotator_class='chunk_entity_resolver', trainable=True, nlu_ref=nlu_ref,
                                is_licensed=True)

        if nlu_ref in ['train.resolve_sentence', 'train.resolve']:
            if not authenticated: print(
                "WARNING! You are trying to train a Licensed Model and your environment does not seem to be Authenticated.\n"
                "Please restart your runtime and run nlu.auth() and for more details see https://nlu.johnsnowlabs.com/docs/en/examples_hc#authorize-access-to-licensed-features-and-install-healthcare-dependencies  ")
            return nlu.Resolver(annotator_class='sentence_entity_resolver', trainable=True, nlu_ref=nlu_ref,
                                is_licensed=True)

        if nlu_ref in ['train.assertion', 'train.assertion_dl']:  # TODO
            return nlu.Classifier(annotator_class='sentiment_dl', trainable=True, nlu_ref=nlu_ref, )



    except:  # if reference is not in namespace and not a component it will cause a unrecoverable crash
        ValueError(
            f'EXCEPTION: Could not create trainable NLU component for nlu_ref = {nlu_ref} and nlp_ref = {nlp_ref}')


def construct_component_from_pipe_identifier(language, nlp_ref, nlu_ref, path=None, is_licensed=False,
                                             strict=False):  # -> NLUPipeline
    '''
    # creates a list of components from a Spark NLP Pipeline reference
    # 1. download pipeline
    # 2. unpack pipeline to annotators and create list of nlu components
    # 3. return list of nlu components
    :param nlu_ref:
    :param language: language of the pipeline
    :param nlp_ref: Reference to a spark nlp petrained pipeline
    :param path: Load pipe from HDD
    :return: Each element of the SaprkNLP pipeline wrapped as a NLU componed inside of a list
    '''

    logger.info("Starting Spark NLP to NLU pipeline conversion process")
    from sparknlp.pretrained import PretrainedPipeline, LightPipeline
    if 'language' in nlp_ref: language = 'xx'  # special edge case for lang detectors
    if path == None:
        if is_licensed:
            pipe = PretrainedPipeline(nlp_ref, lang=language, remote_loc='clinical/models')
        else:
            pipe = PretrainedPipeline(nlp_ref, lang=language)
        iterable_stages = pipe.light_model.pipeline_model.stages
    else:
        pipe = LightPipeline(PipelineModel.load(path=path))
        iterable_stages = pipe.pipeline_model.stages
    constructed_components = []
    # for component in pipe.light_model.pipeline_model.stages:
    for component in iterable_stages:
        logger.info(f"Extracting model from Spark NLP pipeline: {component} and creating Component")
        parsed = str(component).split('_')[0].lower()
        logger.info(f"Parsed Component for : {parsed}")
        if isinstance(component, NerConverter):
            constructed_components.append(
                Util(annotator_class='ner_converter', model=component, lang=language, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                     loaded_from_pretrained_pipe=True))
        elif isinstance(component, MultiClassifierDLModel):
            constructed_components.append(
                nlu.Classifier(model=component, annotator_class='multi_classifier', language=language, nlu_ref=nlu_ref,
                               nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, PerceptronModel):
            constructed_components.append(
                nlu.Classifier(annotator_class='pos', model=component, language=language, nlu_ref=nlu_ref,
                               nlp_ref=nlp_ref,
                               loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, (ClassifierDl, ClassifierDLModel)):
            constructed_components.append(
                nlu.Classifier(annotator_class='classifier_dl', model=component, language=language, nlu_ref=nlu_ref,
                               nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, UniversalSentenceEncoder):
            constructed_components.append(
                nlu.Embeddings(model=component, annotator_class='use', lang=language, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                               loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, BertEmbeddings):
            constructed_components.append(
                nlu.Embeddings(model=component, annotator_class='bert', lang=language, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                               loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, AlbertEmbeddings):
            constructed_components.append(
                nlu.Embeddings(model=component, annotator_class='albert', lang=language, nlu_ref=nlu_ref,
                               nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, XlnetEmbeddings):
            constructed_components.append(
                nlu.Embeddings(model=component, annotator_class='xlnet', lang=language, nlu_ref=nlu_ref,
                               nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, WordEmbeddingsModel):
            constructed_components.append(
                nlu.Embeddings(model=component, annotator_class='glove', lang=language, nlu_ref=nlu_ref,
                               nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, ElmoEmbeddings):
            constructed_components.append(
                nlu.Embeddings(model=component, annotator_class='elmo', lang=language, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                               loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, BertSentenceEmbeddings):
            constructed_components.append(
                nlu.Embeddings(model=component, annotator_class='sentence_bert', lang=language, nlu_ref=nlu_ref,
                               nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, TokenizerModel):
            constructed_components.append(
                nlu.Nlu_Tokenizer(model=component, annotator_class='default_tokenizer', lang=language, nlu_ref=nlu_ref,
                                  nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, DocumentAssembler):
            constructed_components.append(nlu.Util(model=component, lang='xx', nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                                                   loaded_from_pretrained_pipe=True))
        elif isinstance(component, SentenceDetectorDLModel):
            constructed_components.append(
                NLUSentenceDetector(annotator_class='deep_sentence_detector', model=component, lang=language,
                                    nlu_ref=nlu_ref, nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, SentenceDetector):
            constructed_components.append(
                NLUSentenceDetector(annotator_class='pragmatic_sentence_detector', model=component, lang=language,
                                    nlu_ref=nlu_ref, nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True))
        elif parsed in Spellbook.classifiers:
            constructed_components.append(
                nlu.Classifier(model=component, language=language, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                               loaded_from_pretrained_pipe=True))
        elif isinstance(component, Chunker):
            constructed_components.append(
                nlu.chunker.Chunker(annotator_class='default_chunker', model=component, lang=language, nlu_ref=nlu_ref,
                                    nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, ChunkEmbeddings):
            constructed_components.append(
                embeddings_chunker.EmbeddingsChunker(annotator_class='chunk_embedder', model=component, lang=language,
                                                     nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                                                     loaded_from_pretrained_pipe=True))
        elif isinstance(component, RegexMatcherModel) or parsed == 'match':
            constructed_components.append(nlu.Matcher(model=component, annotator_class='regex', nlu_ref=nlu_ref,
                                                      loaded_from_pretrained_pipe=True))
        elif isinstance(component, TextMatcherModel):
            constructed_components.append(
                nlu.Matcher(model=component, annotator_class='text', nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, DateMatcher):
            constructed_components.append(
                nlu.Matcher(model=component, annotator_class='date', nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, ContextSpellCheckerModel):
            constructed_components.append(nlu.SpellChecker(model=component, annotator_class='context', nlu_ref=nlu_ref,
                                                           loaded_from_pretrained_pipe=True))
        elif isinstance(component, SymmetricDeleteModel):
            constructed_components.append(
                nlu.SpellChecker(model=component, annotator_class='symmetric', nlu_ref=nlu_ref,
                                 loaded_from_pretrained_pipe=True))
        elif isinstance(component, NorvigSweetingModel):
            constructed_components.append(nlu.SpellChecker(model=component, annotator_class='norvig_spell', nlu_ref=nlu_ref,
                                                           loaded_from_pretrained_pipe=True))
        elif isinstance(component, LemmatizerModel):
            constructed_components.append(
                nlu.lemmatizer.Lemmatizer(model=component, nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, NormalizerModel):
            constructed_components.append(
                nlu.normalizer.Normalizer(model=component, nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, Stemmer):
            constructed_components.append(
                nlu.stemmer.Stemmer(model=component, nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, (NerDLModel, NerCrfModel)):
            constructed_components.append(
                nlu.Classifier(model=component, annotator_class='ner', language=language, nlu_ref=nlu_ref,
                               nlp_ref=nlp_ref,
                               loaded_from_pretrained_pipe=True))
        elif isinstance(component, LanguageDetectorDL):
            constructed_components.append(
                nlu.Classifier(model=component, annotator_class='language_detector', nlu_ref=nlu_ref,
                               loaded_from_pretrained_pipe=True))
        elif isinstance(component, DependencyParserModel):
            constructed_components.append(
                UnlabledDepParser(model=component, nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, Doc2VecModel):
            constructed_components.append(
                nlu.Embeddings(model=component, annotator_class='doc2vec', lang=language, nlu_ref=nlu_ref,
                               nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True, do_ref_checks=False))
        elif isinstance(component, TypedDependencyParserModel):
            constructed_components.append(
                LabledDepParser(model=component, nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True, ))
        elif isinstance(component, MultiClassifierDLModel):
            constructed_components.append(nlu.Classifier(model=component, nlp_ref='multiclassifierdl', nlu_ref=nlu_ref,
                                                         loaded_from_pretrained_pipe=True))
        elif isinstance(component, (SentimentDetectorModel, SentimentDLModel)):
            constructed_components.append(nlu.Classifier(model=component, nlp_ref='sentimentdl', nlu_ref=nlu_ref,
                                                         loaded_from_pretrained_pipe=True))
        elif isinstance(component, (SentimentDetectorModel, ViveknSentimentModel)):
            constructed_components.append(
                nlu.Classifier(model=component, nlp_ref='vivekn', nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True))

        elif isinstance(component, NGram):
            constructed_components.append(
                nlu.chunker.Chunker(annotator_class='ngram', model=component, nlu_ref=nlu_ref,
                                    loaded_from_pretrained_pipe=True))
        elif isinstance(component, StopWordsCleaner):
            from nlu.components.stopwordscleaner import StopWordsCleaner as Stopw
            constructed_components.append(Stopw(annotator_class='Stopw', model=component, nlu_ref=nlu_ref))
        elif isinstance(component,
                        (TextMatcherModel, RegexMatcherModel, DateMatcher, MultiDateMatcher)) or parsed == 'match':
            constructed_components.append(
                nlu.Matcher(model=component, nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, (T5Transformer)):
            constructed_components.append(
                nlu.Seq2Seq(annotator_class='t5', model=component, nlu_ref=nlu_ref, loaded_from_pretrained_pipe=True))
        elif isinstance(component, (MarianTransformer)):
            constructed_components.append(nlu.Seq2Seq(annotator_class='marian', model=component, nlu_ref=nlu_ref,
                                                      loaded_from_pretrained_pipe=True))

        elif isinstance(component, SentenceEmbeddings):
            constructed_components.append(Util(annotator_class='sentence_embeddings', model=component, nlu_ref=nlu_ref,
                                               loaded_from_pretrained_pipe=True))
        elif parsed in Spellbook.word_embeddings + Spellbook.sentence_embeddings:
            constructed_components.append(
                nlu.Embeddings(model=component, lang=language, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                               loaded_from_pretrained_pipe=True))
        elif isinstance(component, (Finisher, EmbeddingsFinisher)):
            continue  # Dont need finishing since nlu handles finishing
        elif is_licensed:
            from sparknlp_jsl.annotator import AssertionDLModel, AssertionFilterer, AssertionLogRegModel, Chunk2Token, \
                ChunkFilterer
            from sparknlp_jsl.annotator import ChunkMergeModel, ContextualParserModel, DeIdentificationModel, \
                DocumentLogRegClassifierModel, DrugNormalizer
            from sparknlp_jsl.annotator import MedicalNerModel, RelationExtractionModel, RelationExtractionDLModel, \
                RENerChunksFilter, SentenceEntityResolverModel
            # todo embelish ChunkFilterer, Chunk2Token, Disambeguiate, DrugNormalizer, RENerChunksfilterer, IOBTager(???)_, ReIdentify,, NERChunker
            if isinstance(component, AssertionLogRegModel):
                constructed_components.append(
                    nlu.Asserter(annotator_class='assertion_dl', model=component, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                                 loaded_from_pretrained_pipe=True))

            elif isinstance(component, AssertionDLModel):
                constructed_components.append(
                    nlu.Asserter(annotator_class='assertion_log_reg', model=component, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                                 loaded_from_pretrained_pipe=True))

            elif isinstance(component, SentenceEntityResolverModel):
                constructed_components.append(
                    nlu.Resolver(annotator_class='sentence_entity_resolver', model=component, nlu_ref=nlu_ref,
                                 nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True))

            elif isinstance(component, RelationExtractionModel):
                constructed_components.append(
                    nlu.Relation(annotator_class='relation_extractor', model=component, nlu_ref=nlu_ref,
                                 nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True))

            elif isinstance(component, RelationExtractionDLModel):
                constructed_components.append(
                    nlu.Relation(annotator_class='relation_extractor_dl', model=component, nlu_ref=nlu_ref,
                                 nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True))

            elif isinstance(component, MedicalNerModel):
                constructed_components.append(
                    nlu.Classifier(annotator_class='ner_healthcare', model=component, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                                   loaded_from_pretrained_pipe=True))
            elif isinstance(component, ChunkMergeModel):
                constructed_components.append(
                    nlu.Util(annotator_class='chunk_merger', model=component, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                             loaded_from_pretrained_pipe=True))
            elif isinstance(component, ContextualParserModel):
                from nlu.components.chunker import Chunker as Chu
                constructed_components.append(
                    Chu(annotator_class='contextual_parser', model=component, nlu_ref=nlu_ref, nlp_ref=nlp_ref,
                        loaded_from_pretrained_pipe=True))
            elif isinstance(component, DeIdentificationModel):
                constructed_components.append(
                    nlu.Deidentification(annotator_class='deidentifier', model=component, nlu_ref=nlu_ref,
                                         nlp_ref=nlp_ref, loaded_from_pretrained_pipe=True))
            else:
                if strict: raise Exception(
                    f"Could not infer component type for lang={language} and nlp_ref={nlp_ref} of type={component} during pipeline conversion ")
                logger.warning(
                    f"Warning: Could not infer component type for lang={language} and nlp_ref={nlp_ref} and model {component} during pipeline conversion, using default type Normalizer")
                constructed_components.append(
                    nlu.normalizer.Normalizer(model=component, loaded_from_pretrained_pipe=True))
        else:
            if strict: raise Exception(
                f"Could not infer component type for lang={language} and nlp_ref={nlp_ref} of type={component} during pipeline conversion ")
            logger.warning(
                f"Warning: Could not infer component type for lang={language} and nlp_ref={nlp_ref} and model {component} during pipeline conversion, using default type Normalizer")
            constructed_components.append(nlu.normalizer.Normalizer(model=component, loaded_from_pretrained_pipe=True))

        logger.info(f"Extracted into NLU Component type : {parsed}", )
        if None in constructed_components: raise Exception(
            f"Could not infer component type for lang={language} and nlp_ref={nlp_ref} during pipeline conversion,")
    return ComponentUtils.set_storage_ref_attribute_of_embedding_converters(
        PipeUtils.set_column_values_on_components_from_pretrained_pipe(constructed_components, nlp_ref, language, path))


def construct_component_from_identifier(language, component_type='', dataset='', component_embeddings='', nlu_ref='',
                                        nlp_ref='', is_licensed=False) -> NLUComponent:
    '''
    Creates a NLU component from a pretrained SparkNLP model reference or Class reference. First step to get the Root of the NLP DAG
    Class references will return default pretrained models
    :param language: Language of the sparknlp model reference
    :param component_type: Class which will be used to instantiate the model
    :param dataset: Dataset that the model was trained on
    :param component_embeddings: Embedded that the models was traiend on (if any)
    :param nlu_ref: Full user request
    :param nlp_ref: Full Spark NLP reference
    :return: Returns a NLU component which embelished the Spark NLP pretrained model and class for that model
    '''
    logger.info(
        f'Creating singular NLU component for type={component_type} sparknlp_ref={nlp_ref} , nlu_ref={nlu_ref} dataset={dataset}, language={language} ')
    try:
        if 'assert' in component_type:
            return nlu.Asserter(nlp_ref=nlp_ref, nlu_ref=nlu_ref, lang=language, get_default=False,
                                is_licensed=is_licensed)
        elif 'resolve' in component_type or 'resolve' in nlu_ref:
            return nlu.Resolver(nlp_ref=nlp_ref, nlu_ref=nlu_ref, language=language, get_default=False,
                                is_licensed=is_licensed)
        elif 'relation' in nlu_ref:
            return nlu.Relation(nlp_ref=nlp_ref, nlu_ref=nlu_ref, lang=language, get_default=False,
                                is_licensed=is_licensed)
        elif 'de_identify' in nlu_ref and 'ner' not in nlu_ref:
            return nlu.Deidentification(nlp_ref=nlp_ref, nlu_ref=nlu_ref, lang=language, get_default=False,
                                        is_licensed=is_licensed)

        elif any(x in Spellbook.seq2seq for x in [nlp_ref, nlu_ref, dataset, component_type, ]):
            return Seq2Seq(annotator_class=component_type, language=language, get_default=False, nlp_ref=nlp_ref,
                           configs=dataset, is_licensed=is_licensed)

        # if any([component_type in NameSpace.word_embeddings,dataset in NameSpace.word_embeddings, nlu_ref in NameSpace.word_embeddings, nlp_ref in NameSpace.word_embeddings]):
        elif any(
                x in Spellbook.classifiers for x in [nlp_ref, nlu_ref, dataset, component_type, ] + dataset.split('_')):
            return Classifier(get_default=False, nlp_ref=nlp_ref, nlu_ref=nlu_ref, language=language,
                              is_licensed=is_licensed)


        elif any(x in Spellbook.word_embeddings and x not in Spellbook.classifiers for x in
                 [nlp_ref, nlu_ref, dataset, component_type, ] + dataset.split('_')):
            return Embeddings(get_default=False, nlp_ref=nlp_ref, nlu_ref=nlu_ref, lang=language,
                              is_licensed=is_licensed)

        # elif any([component_type in NameSpace.sentence_embeddings,dataset in NameSpace.sentence_embeddings, nlu_ref in NameSpace.sentence_embeddings, nlp_ref in NameSpace.sentence_embeddings]):
        if any(x in Spellbook.sentence_embeddings and not x in Spellbook.classifiers for x in
               [nlp_ref, nlu_ref, dataset, component_type, ] + dataset.split('_')):
            return Embeddings(get_default=False, nlp_ref=nlp_ref, nlu_ref=nlu_ref, lang=language,
                              is_licensed=is_licensed)

        if any(x in Spellbook.token_classifiers and not x in Spellbook.classifiers for x in
               [nlp_ref, nlu_ref, dataset, component_type, ] + dataset.split('_')):
            return Embeddings(get_default=False, nlp_ref=nlp_ref, nlu_ref=nlu_ref, lang=language,
                              is_licensed=is_licensed)

        elif any('spell' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return SpellChecker(annotator_class=component_type, language=language, get_default=True, nlp_ref=nlp_ref,
                                nlu_ref=nlu_ref,
                                dataset=dataset, is_licensed=is_licensed)

        elif any('dep' in x and not 'untyped' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return LabledDepParser()

        elif any('dep.untyped' in x or 'untyped' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return UnlabledDepParser()

        elif any('lemma' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return nlu.lemmatizer.Lemmatizer(language=language, nlp_ref=nlp_ref, is_licensed=is_licensed)

        elif any('norm' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return nlu.normalizer.Normalizer(nlp_ref=nlp_ref, nlu_ref=nlu_ref, is_licensed=is_licensed)
        elif any('clean' in x or 'stopword' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return nlu.StopWordsCleaners(lang=language, get_default=False, nlp_ref=nlp_ref)
        elif any('sentence_detector' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return NLUSentenceDetector(nlu_ref=nlu_ref, nlp_ref=nlp_ref, language=language, is_licensed=is_licensed)

        elif any('match' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return Matcher(nlu_ref=nlu_ref, nlp_ref=nlp_ref, is_licensed=is_licensed)


        elif any('tokenize' in x or 'segment_words' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return nlu.nlu_tokenizer.Nlu_Tokenizer(nlp_ref=nlp_ref, nlu_ref=nlu_ref, language=language,
                                                   get_default=False,
                                                   is_licensed=is_licensed)

        elif any('stem' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return nlu.Stemmers()

        # supported in future version with auto embed generation
        # elif any('embed_chunk' in x for x in [nlp_ref, nlu_ref, dataset, component_type] ):
        #     return embeddings_chunker.EmbeddingsChunker()    DrugNormalizer,

        elif any('chunk' in x for x in [nlp_ref, nlu_ref, dataset, component_type]):
            return nlu.chunker.Chunker()
        elif component_type == 'ngram':
            return nlu.chunker.Chunker('ngram')

        logger.exception(
            f'EXCEPTION: Could not resolve singular Component for type={component_type} and nlp_ref={nlp_ref} and nlu_ref={nlu_ref} and lang ={language} ')
        return None
        # raise ValueError
    except:  # if reference is not in namespace and not a component it will cause a unrecoverable crash
        logger.exception(
            f'EXCEPTION: Could not resolve singular Component for type={component_type} and nlp_ref={nlp_ref} and nlu_ref={nlu_ref} and lang ={language} ')
        return None
        # raise ValueError
