import logging

import nlu
from nlu import Licenses
from nlu.info import AllComponentsInfo
from nlu.spellbook import Spellbook

logger = logging.getLogger('nlu')


def check_if_nlu_ref_is_licensed(nlu_ref):
    """check if a nlu_ref is pointing to a licensed or open source model_anno_obj.
    This works by just checking if the NLU ref points to a healthcare model_anno_obj or not"""
    for lang, universe in Spellbook.healthcare_component_alias_references.items():
        for hc_nlu_ref, hc_nlp_ref in universe.items():
            if hc_nlu_ref == nlu_ref: return True
    for lang, universe in Spellbook.pretrained_healthcare_model_references.items():
        for hc_nlu_ref, hc_nlp_ref in universe.items():
            if hc_nlu_ref == nlu_ref: return True
    return False


def parse_language_from_nlu_ref(nlu_ref):
    """Parse a ISO language identifier from a NLU reference which can be used to load a Spark NLP model_anno_obj"""
    infos = nlu_ref.split('.')
    lang = None
    for split in infos:
        if split in nlu.Spellbook.pretrained_models_references.keys():
            lang = split

    if not lang:
        # translators are internally handled as 'xx'
        if 'translate_to' in nlu_ref and not 't5' in nlu_ref:
            # Special case for translate_to, will be prefixed with 'xx' if not already prefixed with xx because there are only xx.<ISO>.translate_to references
            language = 'xx'
            if nlu_ref[0:3] != 'xx.':
                nlu_reference = 'xx.' + nlu_ref
            logger.info(f'Setting lang as xx for nlu_ref={nlu_reference}')
    if not lang:
        lang = 'en'
    logger.info(f'Parsed Nlu_ref={nlu_ref} as lang={lang}')

    return lang


def nlu_ref_to_nlp_ref(nlu_ref):
    _, _, nlp_ref, _, _ = nlu_ref_to_nlp_metadata(nlu_ref)
    return nlp_ref


def nlu_ref_to_nlp_metadata(nlu_ref, is_recursive_call=False):
    """
    For given NLU ref, returns is_pipe, license_type

    :return: lang, nlu_ref, nlp_ref, license_type, is_pipe
    """
    model_params = None
    lang = parse_language_from_nlu_ref(nlu_ref)
    nlp_ref = None
    license_type = Licenses.open_source
    is_pipe = False
    if 'translate_to' in nlu_ref :
        # We append here xx and set lang as xx  so users don't have to specify it
        lang = 'xx'
        if 'xx' not in nlu_ref:
            nlu_ref = 'xx.' + nlu_ref
    # 1. check if open source pipeline
    if lang in Spellbook.pretrained_pipe_references.keys():
        if nlu_ref in Spellbook.pretrained_pipe_references[lang].keys():
            nlp_ref = Spellbook.pretrained_pipe_references[lang][nlu_ref]
            is_pipe = True
    # 2. check if open source model_anno_obj
    if lang in Spellbook.pretrained_models_references.keys():
        if nlu_ref in Spellbook.pretrained_models_references[lang].keys():
            nlp_ref = Spellbook.pretrained_models_references[lang][nlu_ref]
            logger.info(f'Found Spark NLP reference in pretrained models namespace = {nlp_ref}')

    # 3. check if open source alias
    if nlu_ref in Spellbook.component_alias_references.keys():
        sparknlp_data = Spellbook.component_alias_references[nlu_ref]
        nlp_ref = sparknlp_data[0]
        is_pipe = 'component_list' in sparknlp_data[1]
        if len(sparknlp_data) == 3:
            model_params = sparknlp_data[2]
    # 4. check if healthcare pipe
    if lang in Spellbook.pretrained_healthcare_pipe_references.keys():
        if nlu_ref in Spellbook.pretrained_healthcare_pipe_references[lang].keys():
            nlp_ref = Spellbook.pretrained_healthcare_pipe_references[lang][nlu_ref]
            license_type = Licenses.hc
            is_pipe = True

    # 5. check if healthcare model_anno_obj
    if lang in Spellbook.pretrained_healthcare_model_references.keys():
        if nlu_ref in Spellbook.pretrained_healthcare_model_references[lang].keys():
            nlp_ref = Spellbook.pretrained_healthcare_model_references[lang][nlu_ref]
            license_type = Licenses.hc

    # 6. check if healthcare alias
    if nlu_ref in Spellbook.healthcare_component_alias_references.keys():
        sparknlp_data = Spellbook.healthcare_component_alias_references[nlu_ref]
        nlp_ref = sparknlp_data[0]
        is_pipe = 'component_list' in sparknlp_data[1]
        license_type = Licenses.hc

    # 7. check if ocr model_anno_obj
    if nlu_ref in Spellbook.ocr_model_references.keys():
        nlp_ref = Spellbook.ocr_model_references[nlu_ref]
        license_type = Licenses.ocr

    # Check if multi lingual ner
    if not nlp_ref and 'ner' in nlu_ref:
        all_component_info = AllComponentsInfo()
        if lang in all_component_info.all_multi_lang_base_ner_languages:
            lang = 'xx'
            nlp_ref = 'ner_wikiner_glove_840B_300'
            nlu_ref = 'xx.ner.wikiner_glove_840B_300'
        if lang in all_component_info.all_multi_lang_xtreme_ner_languages:
            lang = 'xx'
            nlp_ref = 'ner_xtreme_glove_840B_300'
            nlu_ref = 'xx.ner.xtreme_glove_840B_300'

    # Search again but with en. prefixed, enables all refs to work without en prefix
    if not nlp_ref and not is_recursive_call:
        return nlu_ref_to_nlp_metadata('en.' + nlu_ref, is_recursive_call=True)
    #
    # if is_licensed:
    #     if not auth_utils.is_authorized_environment() and is_licensed:
    #         print(f"The nlu_ref=[{nlu_ref}] is pointing to a licensed Spark NLP Annotator or Model [{nlp_ref}]. \n"
    #               f"Your environment does not seem to be Authorized!\n"
    #               f"Please RESTART your Python environment and run nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)  \n"
    #               f"with your corrosponding credentials. If you have no credentials you can get a trial version with credentials here https://www.johnsnowlabs.com/spark-nlp-try-free/ \n"
    #               f"Or contact us at contact@jonsnowlabs.com\n"
    #               f"NLU will ignore this error and continue running, but you will encounter errors most likely. ")

    return lang, nlu_ref, nlp_ref, license_type, is_pipe, model_params
