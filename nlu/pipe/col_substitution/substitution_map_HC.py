"""
Resolve Annotator Classes in the Pipeline to Extractor Configs and Methods

Every Annotator should have 2 configs. Some might offor multuple configs/method pairs, based on model/NLP reference.
- default/minimalistic -> Just the results of the annotations, no confidences or extra metadata
- with meta            -> A config that leverages white/black list and gets the most relevant metadata
- with positions       -> With Begins/Ends
- with sentence references -> Reeturn the sentence/chunk no. reference from the metadata.
                                If a document has multi-sentences, this will map a label back to a corrosponding sentence

"""
from nlu.pipe.col_substitution.col_substitution_HC import *
from nlu.pipe.col_substitution.col_substitution_OS import *

from sparknlp_jsl.annotator  import *

HC_anno2substitution_fn = {
    MedicalNerModel : {
        'default': substitute_ner_dl_cols ,
    },
    NerConverterInternal : {
        'default': substitute_ner_internal_converter_cols,
    },
    AssertionDLModel : {
        'default': substitute_assertion_cols,
    },
    AssertionLogRegModel : {
        'default': substitute_assertion_cols,
    },
    SentenceEntityResolverModel : {
        'default': substitute_sentence_resolution_cols,
    },
    ChunkEntityResolverModel : {
        'default': substitute_chunk_resolution_cols,
    },
    ChunkEntityResolverApproach : {
        'default': substitute_chunk_resolution_cols,
    },


    DeIdentificationModel : {
        'default': substitute_de_identification_cols,
    },
    RelationExtractionModel : {
        'default': substitute_relation_cols,
    },

    RelationExtractionDLModel : {
        'default': substitute_relation_cols,
    },
    Chunk2Token : {
        'default': '',# TODO
    },

    ContextualParserApproach : {
        'default' :substitute_context_parser_cols,#

    },
    ContextualParserModel : {
        'default': substitute_context_parser_cols,

    },

    DrugNormalizer : {
        'default': substitute_drug_normalizer_cols
    },

    GenericClassifierModel : {
        'default': substitute_generic_classifier_parser_cols,
    },
    GenericClassifierApproach : {
        'default': substitute_generic_classifier_parser_cols,
    },


    ChunkMergeModel : {
        'default': '',# TODO
    },

    NerDisambiguatorModel : {
        'default': '',# TODO
    },

    RENerChunksFilter : {
        'default': '',# TODO
    },

    NerOverwriter : {
        'default': '',# TODO
    },
    PosologyREModel : {
        'default': substitute_relation_cols,
    }

}







