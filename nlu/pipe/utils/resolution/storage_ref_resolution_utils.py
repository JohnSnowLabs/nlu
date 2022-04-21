import logging
from nlu.pipe.utils.resolution.nlu_ref_utils import *
from nlu.spellbook import Spellbook
from nlu.universe.feature_universes import NLP_FEATURES

logger = logging.getLogger('nlu')


def resolve_storage_ref(lang, storage_ref, missing_component_type):
    """Returns a nlp_ref, nlu_ref and whether it is a licensed model_anno_obj or not and an updated languiage, if multi lingual"""
    logger.info(
        f"Resolving storage_ref={storage_ref} for lang={lang} and missing_component_type={missing_component_type}")
    nlu_ref, nlp_ref, is_licensed = None, None, False
    # get nlu ref

    # check if storage_ref is hardcoded
    if lang in Spellbook.licensed_storage_ref_2_nlu_ref.keys() and storage_ref in \
            Spellbook.licensed_storage_ref_2_nlu_ref[lang].keys():
        nlu_ref = Spellbook.licensed_storage_ref_2_nlu_ref[lang][storage_ref]
        is_licensed = True
    elif lang in Spellbook.storage_ref_2_nlu_ref.keys() and storage_ref in Spellbook.storage_ref_2_nlu_ref[
        lang].keys():
        nlu_ref = Spellbook.storage_ref_2_nlu_ref[lang][
            storage_ref]  # a HC model_anno_obj may use OS storage_ref_provider, so we dont know yet if it is licensed or not
    if lang in Spellbook.pretrained_models_references.keys() and nlu_ref in \
            Spellbook.pretrained_models_references[lang].keys():
        nlp_ref = Spellbook.pretrained_models_references[lang][nlu_ref]
    elif lang in Spellbook.pretrained_healthcare_model_references.keys() and nlu_ref in \
            Spellbook.pretrained_healthcare_model_references[lang].keys():
        nlp_ref = Spellbook.pretrained_healthcare_model_references[lang][nlu_ref]
        is_licensed = True
    # check if storage_ref matches nlu_ref and get NLP_ref
    elif lang in Spellbook.licensed_storage_ref_2_nlu_ref.keys() and storage_ref in \
            Spellbook.licensed_storage_ref_2_nlu_ref[lang].keys():
        nlu_ref = storage_ref
        nlp_ref = Spellbook.licensed_storage_ref_2_nlu_ref[lang][nlu_ref]
    elif lang in Spellbook.pretrained_models_references.keys() and storage_ref in \
            Spellbook.pretrained_models_references[lang].keys():
        nlu_ref = storage_ref
        nlp_ref = Spellbook.pretrained_models_references[lang][nlu_ref]

    # check if storage_ref matches nlp_ref and get nlp and nlu ref
    elif lang in Spellbook.pretrained_healthcare_model_references.keys():
        if storage_ref in Spellbook.pretrained_healthcare_model_references[lang].values():
            inv_namespace = {v: k for k, v in Spellbook.pretrained_healthcare_model_references[lang].items()}
            nlp_ref = storage_ref
            nlu_ref = inv_namespace[nlp_ref]
            is_licensed = True

    if nlu_ref is not None and 'xx.' in nlu_ref: lang = 'xx'

    if nlp_ref is None and nlu_ref is not None:
        # cast NLU ref to NLP ref
        if is_licensed:
            nlp_ref = Spellbook.pretrained_healthcare_model_references[lang][nlu_ref]
        else:
            nlp_ref = Spellbook.pretrained_models_references[lang][nlu_ref]

    if nlp_ref is not None and nlu_ref is None:
        # cast NLP ref to NLU ref
        if is_licensed:
            inv_namespace = {v: k for k, v in Spellbook.pretrained_healthcare_model_references[lang].items()}
            nlu_ref = inv_namespace[nlp_ref]
        else:
            inv_namespace = {v: k for k, v in Spellbook.pretrained_models_references[lang].items()}
            nlu_ref = inv_namespace[nlp_ref]

    if nlu_ref == None and nlp_ref == None:
        # todo enfore storage ref when training
        logger.info(f"COULD NOT RESOLVE STORAGE_REF={storage_ref}")
        if storage_ref == '':
            if missing_component_type == NLP_FEATURES.SENTENCE_EMBEDDINGS:
                logger.info("Using default storage_ref USE, assuming training mode")
                storage_ref = 'en.embed_sentence.use'  # this enables default USE embeds for traianble components
                nlp_ref = 'tfhub_use'
                nlu_ref = storage_ref
            elif missing_component_type == NLP_FEATURES.WORD_EMBEDDINGS:
                logger.info("Using default storage_ref GLOVE, assuming training mode")
                storage_ref = 'en.glove'  # this enables default USE embeds for traianble components
                nlp_ref = 'glove_100d'
                nlu_ref = storage_ref
        else:
            nlp_ref = storage_ref
            nlu_ref = storage_ref
    if nlu_ref is not None:
        is_licensed = check_if_nlu_ref_is_licensed(nlu_ref)

    logger.info(f'Resolved storageref = {storage_ref} to NLU_ref = {nlu_ref} and NLP_ref = {nlp_ref}')
    return nlu_ref, nlp_ref, is_licensed, lang


def set_storage_ref_and_resolution_on_component_info(c, storage_ref):
    """Sets a storage ref on a components component_to_resolve info and returns the component_to_resolve """
    c.storage_ref = storage_ref
    return c
