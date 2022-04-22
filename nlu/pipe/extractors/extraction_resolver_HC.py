"""
Resolve Annotator Classes in the Pipeline to Extractor Configs and Methods

Every Annotator should have 2 configs. Some might offor multuple configs/method pairs, based on model_anno_obj/NLP reference.
- default/minimalistic -> Just the results of the annotations, no confidences or extra metadata
- with meta            -> A config that leverages white/black list and gets the most relevant metadata
- with positions       -> With Begins/Ends
- with sentence references -> Reeturn the sentence/chunk no. reference from the metadata.
                                If a document has multi-sentences, this will map a label back to a corrosponding sentence

"""
from nlu.pipe.extractors.extractor_configs_OS import *
from nlu.pipe.extractors.extractor_configs_HC import *

from sparknlp_jsl.annotator  import *
from sparknlp_jsl.base import *

HC_anno2config = {
    MedicalNerModel : {
        'default': default_ner_config,
        # 'meta': meta_NER_config,
        'default_full'  : default_full_config,
    },
    NerConverterInternal : {
        'default': default_NER_converter_licensed_config,
        'default_full'  : default_full_config,
    },
    AssertionDLModel : {
        'default': default_assertion_config,
        'default_full'  : default_full_config,
    },
    AssertionLogRegModel : {
        'default': default_assertion_config,
        'default_full'  : default_full_config,
    },
    SentenceEntityResolverModel : {
        'default': default_chunk_resolution_config,
        'default_full'  : default_full_config,
    },
    SentenceEntityResolverApproach : {
        'default': default_chunk_resolution_config,
        'default_full'  : default_full_config,
    },
    # ChunkEntityResolverModel : {
    #     'default': default_chunk_resolution_config,
    #     'default_full'  : default_full_config,
    # },
    # ChunkEntityResolverApproach : {
    #     'default': default_chunk_resolution_config,
    #     'default_full'  : default_full_config,
    # },

    DeIdentificationModel : {
        'default': default_de_identification_config,
        'default_full'  : default_full_config,
    },
    RelationExtractionModel : {
        'default': default_relation_extraction_config,
        'positional': default_relation_extraction_positional_config,
        'default_full'  : default_full_config,
    },

    RelationExtractionDLModel : {
        'default': default_relation_extraction_config,
        'positional': default_relation_extraction_positional_config,
        'default_full'  : default_full_config,
    },
    Chunk2Token : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },

    ContextualParserModel : {
        'default': default_full_config,# TODO
        'default_full'  : default_full_config,

    },

    ContextualParserApproach : {
        'default': default_full_config,# TODO
        'default_full'  : default_full_config,

    },
    DrugNormalizer : {
        'default': default_only_result_config,
        'default_full'  : default_full_config,
    },

    GenericClassifierModel : {
        'default': default_generic_classifier_config,
        'default_full'  : default_full_config,
    },

    GenericClassifierApproach : {
        'default': default_generic_classifier_config,
        'default_full'  : default_full_config,
    },


    FeaturesAssembler : {
        'default': default_feature_assembler_config,
        'default_full'  : default_full_config,
    },


    ChunkMergeModel : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },

    NerDisambiguatorModel : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },

    RENerChunksFilter : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },

    NerOverwriter : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },
    PosologyREModel : {
        # 'default': '',# TODO
        'default': default_relation_extraction_positional_config,
        'default_full'  : default_full_config,

    }

}







