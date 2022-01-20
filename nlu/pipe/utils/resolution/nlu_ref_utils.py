from nlu.spellbook import Spellbook
import logging
logger = logging.getLogger('nlu')

from nlu.info import AllComponentsInfo

def check_if_nlu_ref_is_licensed(nlu_ref):
    """check if a nlu_ref is pointing to a licensed or open source model.
    This works by just checking if the NLU ref points to a healthcare model or not"""
    for lang, universe in Spellbook.healthcare_component_alias_references.items():
        for hc_nlu_ref, hc_nlp_ref in universe.items():
            if hc_nlu_ref == nlu_ref: return True
    for lang, universe in Spellbook.pretrained_healthcare_model_references.items():
        for hc_nlu_ref, hc_nlp_ref in universe.items():
            if hc_nlu_ref == nlu_ref: return True
    return False

def parse_language_from_nlu_ref(nlu_ref):
    """Parse a ISO language identifier from a NLU reference which can be used to load a Spark NLP model"""
    infos = nlu_ref.split('.')
    for split in infos:
        if split in AllComponentsInfo().all_languages:
            logger.info(f'Parsed Nlu_ref={nlu_ref} as lang={split}')
            return split
    logger.info(f'Parsed Nlu_ref={nlu_ref} as lang=en')
    return 'en'


def extract_classifier_metadata_from_nlu_ref(nlu_ref):
    '''
    Extract classifier and metadataname from nlu reference which is handy for deciding what output column names should be
    Strips lang and action from nlu_ref and returns a list of remaining identifiers, i.e [<classifier_name>,<classifier_dataset>, <additional_classifier_meta>
    :param nlu_ref: nlu reference from which to extra model meta data
    :return: [<modelname>, <dataset>, <more_meta>,]  . For pure actions this will return []
    '''
    model_infos = []
    for e in nlu_ref.split('.'):
        if e in AllComponentsInfo().all_languages or e in Spellbook.actions: continue
        model_infos.append(e)
    return model_infos