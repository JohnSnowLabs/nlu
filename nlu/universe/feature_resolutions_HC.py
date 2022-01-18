"""
Collection of universes shared across all libraries (NLP/HC/OCR), which are collections of atoms
"""
from dataclasses import dataclass

from nlu.pipe.nlu_component import NluComponent
from nlu.universe.component_universes import ComponentMap
from nlu.universe.feature_node_ids import NLP_NODE_IDS, NLP_HC_NODE_IDS
from nlu.universe.feature_node_universes import NLP_FEATURE_NODES
from nlu.universe.feature_universes import NLP_FEATURES


### ____ Annotator Feature Representations ____


@dataclass
class NlpHcFeatureResolutions:
    default_HC_resolutions = {
        NLP_FEATURES.NAMED_ENTITY_CONVERTED: ComponentMap.hc_components[NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL],
        # jsl_ner_wip_clinical
        NLP_FEATURES.NAMED_ENTITY_IOB: ComponentMap.hc_components[NLP_HC_NODE_IDS.MEDICAL_NER]
    }

    default_HC_train_resolutions = {

        NLP_FEATURES.NAMED_ENTITY_CONVERTED:  ComponentMap.os_components[NLP_NODE_IDS.DOC2CHUNK]
    }
    # speed_optimized_resolutions: Dict[JslFeature,(JslAnnoId, NluRef)] = None # todo for nlp expert


@dataclass
class ResolvedFeature:
    nlu_ref: str
    nlp_ref: str
    language: str
    get_pretrained: bool  # Call get_pretrained(nlp_ref, lang, bucket) or get_default() on the AnnotatorClass
    nlu_component: NluComponent


