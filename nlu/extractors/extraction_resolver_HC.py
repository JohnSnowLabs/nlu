"""
Resolve Annotator Classes in the Pipeline to Extractor Configs and Methods

Every Annotator should have 2 configs. Some might offor multuple configs/method pairs, based on model/NLP reference.
- default/minimalistic -> Just the results of the annotations, no confidences or extra metadata
- with meta            -> A config that leverages white/black list and gets the most relevant metadata
- with positions       -> With Begins/Ends
- with sentence references -> Reeturn the sentence/chunk no. reference from the metadata.
                                If a document has multi-sentences, this will map a label back to a corrosponding sentence

"""
from nlu.extractors.extractor_configs_open_source import *
from nlu.extractors.extractor_configs_healthcare import *

from sparknlp_jsl.annotator  import *
# Todo one dict for Spark2 and one for Spark3, because medical NER will give errors when importing in Spark 2

HC_anno2config = {
    MedicalNerModel : {
        'default': default_ner_config,
        'meta': meta_NER_config,
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
        'default': default_chunk_resolution_config,
        'default_full'  : default_full_config,
    },
    ChunkEntityResolverModel : {
        'default': default_chunk_resolution_config,
        'default_full'  : default_full_config,
        # Todo instead of topK return Best and pre-rpcoess results, i.e. splti on :::
    },

    DeIdentificationModel : {
        'default': default_de_identification_config,
        'default_full'  : default_full_config,
    },
    RelationExtractionModel : {
        'default': default_relation_extraction_positional_config,
        'positional': positional_relation_extraction_config,
        'default_full'  : default_full_config,
    },

    RelationExtractionDLModel : {
        'default': default_relation_extraction_positional_config,
        'positional': positional_relation_extraction_config,
        'default_full'  : default_full_config,
    },
    Chunk2Token : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },

    ContextualParserModel : {
        'default': '',# TODO
        'default_full'  : default_full_config,

    },

    DrugNormalizer : {
        'default': '',# TODO
        'default_full'  : default_full_config,
    },

    GenericClassifierModel : {
        'default': '',# TODO
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







