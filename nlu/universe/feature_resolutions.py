"""
Collection of universes shared across all libraries (NLP/HC/OCR), which are collections of atoms
"""
from dataclasses import dataclass

from nlu.pipe.nlu_component import NluComponent
from nlu.universe.component_universes import ComponentMap
from nlu.universe.feature_node_ids import NLP_NODE_IDS, NLP_HC_NODE_IDS, OCR_NODE_IDS
from nlu.universe.feature_node_universes import NLP_FEATURE_NODES
from nlu.universe.feature_universes import NLP_FEATURES, OCR_FEATURES


### ____ Annotator Feature Representations ____

@dataclass
class OcrFeatureResolutions:
    resolutions = {NLP_FEATURES.DOCUMENT: NLP_FEATURE_NODES.nodes[NLP_NODE_IDS.DOCUMENT_ASSEMBLER]}


@dataclass
class NlpHcFeatureResolutions:
    resolutions = {NLP_FEATURES.DOCUMENT: NLP_FEATURE_NODES.nodes[NLP_NODE_IDS.DOCUMENT_ASSEMBLER]}


@dataclass
class ResolvedFeature:
    nlu_ref: str
    nlp_ref: str
    language: str
    get_pretrained: bool  # Call get_pretrained(nlp_ref, lang, bucket) or get_default() on the AnnotatorClass
    nlu_component: NluComponent # Resolving component

class NlpFeatureResolutions:
    # Map each requested Feature to a pre-defined optimal resolution, given by FeatureNode
    # TODO some features should be NLP-EXPERT config dependent!!! Especially for training
    # TODO we need ideal resolution for each lang...!
    # Also We need Alternative Default wether licensed or not!!
    # Ideally we define nlu_ref for each of these
    # default_resolutions: Dict[JslFeature,JslAnnoId] = None
    # TODO use lang families, i.e. en.tokenize works for all Latin style languages but not Chinese, I.e. not actually multi lingual

    default_OS_resolutions = {
        NLP_FEATURES.DOCUMENT: ResolvedFeature('document_assembler', 'document_assembler', 'xx', False,
                                               ComponentMap.os_components[NLP_NODE_IDS.DOCUMENT_ASSEMBLER]),
        NLP_FEATURES.TOKEN: ResolvedFeature('en.tokenize', 'spark_nlp_tokenizer', 'en', False,
                                            ComponentMap.os_components[NLP_NODE_IDS.TOKENIZER]),

        NLP_FEATURES.SENTENCE: ResolvedFeature('detect_sentence', 'sentence_detector_dl', 'en', False,
                                               ComponentMap.os_components[NLP_NODE_IDS.SENTENCE_DETECTOR_DL]),
        NLP_FEATURES.SENTENCE_EMBEDDINGS: ResolvedFeature('en.embed_sentence.small_bert_L2_128',
                                                          'sent_small_bert_L2_128', 'en', True,
                                                          ComponentMap.os_components[
                                                              NLP_NODE_IDS.BERT_SENTENCE_EMBEDDINGS]),
        NLP_FEATURES.WORD_EMBEDDINGS: ResolvedFeature('en.embed.bert.small_L2_128', 'small_bert_L2_128', 'en', True,
                                                      ComponentMap.os_components[NLP_NODE_IDS.BERT_EMBEDDINGS]),
        NLP_FEATURES.POS: ResolvedFeature('en.pos', 'pos_anc', 'en', True,
                                          ComponentMap.os_components[NLP_NODE_IDS.POS]),
        NLP_FEATURES.NAMED_ENTITY_IOB: ResolvedFeature('en.ner.onto.bert.cased_base', 'onto_bert_base_cased', 'en',
                                                       True,
                                                       ComponentMap.os_components[NLP_NODE_IDS.NER_DL]),
        NLP_FEATURES.CHUNK_EMBEDDINGS: ResolvedFeature('en.embed_chunk', 'chunk_embeddings', 'xx', False,
                                                       ComponentMap.os_components[
                                                           NLP_NODE_IDS.CHUNK_EMBEDDINGS_CONVERTER]),

        NLP_FEATURES.NAMED_ENTITY_CONVERTED: ResolvedFeature('ner_converter', 'ner_converter', 'xx', False,
                                                             ComponentMap.os_components[NLP_NODE_IDS.NER_CONVERTER]),
        NLP_FEATURES.LABELED_DEPENDENCY: ResolvedFeature('en.dep.untyped', 'dependency_conllu', 'en', True,
                                                         ComponentMap.os_components[
                                                             NLP_NODE_IDS.TYPED_DEPENDENCY_PARSER]),
        NLP_FEATURES.UNLABLED_DEPENDENCY: ResolvedFeature('en.dep.untyped', 'dependency_conllu', 'en', True,
                                                          ComponentMap.os_components[
                                                             NLP_NODE_IDS.UNTYPED_DEPENDENCY_PARSER]),
        NLP_FEATURES.CHUNK: ResolvedFeature('en.chunk', 'default_chunker', 'xx', False,
                                            ComponentMap.os_components[NLP_NODE_IDS.CHUNKER]),

        NLP_FEATURES.DOCUMENT_FROM_CHUNK: ResolvedFeature(NLP_NODE_IDS.DOC2CHUNK, NLP_NODE_IDS.DOC2CHUNK, 'xx', False,
                                            ComponentMap.os_components[NLP_NODE_IDS.DOC2CHUNK]),


    }

    default_HC_resolutions = {
        NLP_FEATURES.NAMED_ENTITY_CONVERTED: ResolvedFeature(NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL, NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL, 'xx', False,
                                                             ComponentMap.hc_components[NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL]),
        NLP_FEATURES.NAMED_ENTITY_IOB: ResolvedFeature('en.med_ner.jsl', 'ner_jsl', 'en',
                                                       True,
                                                       ComponentMap.hc_components[NLP_HC_NODE_IDS.MEDICAL_NER]),
    }

    default_HC_train_resolutions = {
        NLP_FEATURES.NAMED_ENTITY_CONVERTED: ResolvedFeature(NLP_NODE_IDS.DOC2CHUNK, NLP_NODE_IDS.DOC2CHUNK, 'xx', False,
                                                             ComponentMap.os_components[NLP_NODE_IDS.DOC2CHUNK]),
    }

    default_OCR_resolutions = {
        OCR_FEATURES.OCR_IMAGE: ResolvedFeature(OCR_NODE_IDS.BINARY2IMAGE, OCR_NODE_IDS.BINARY2IMAGE, 'xx', False,
                                                ComponentMap.ocr_components[OCR_NODE_IDS.BINARY2IMAGE]),
    }
