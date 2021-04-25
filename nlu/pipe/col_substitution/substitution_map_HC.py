"""
Resolve Annotator Classes in the Pipeline to Extractor Configs and Methods

Every Annotator should have 2 configs. Some might offor multuple configs/method pairs, based on model/NLP reference.
- default/minimalistic -> Just the results of the annotations, no confidences or extra metadata
- with meta            -> A config that leverages white/black list and gets the most relevant metadata
- with positions       -> With Begins/Ends
- with sentence references -> Reeturn the sentence/chunk no. reference from the metadata.
                                If a document has multi-sentences, this will map a label back to a corrosponding sentence

"""
from nlu.pipe.extractors.extractor_configs_open_source import *
from nlu.pipe.extractors.extractor_configs_healthcare import *

from sparknlp_jsl.annotator  import *

HC_anno2substitution_fn = {
    MedicalNerModel : {
        'default': 'TODO',
    },
    NerConverterInternal : {
        'default': 'TODO',
    },
    AssertionDLModel : {
        'default': 'TODO',
    },
    AssertionLogRegModel : {
        'default': 'TODO',
    },
    SentenceEntityResolverModel : {
        'default': 'TODO',
    },
    ChunkEntityResolverModel : {
        'default': 'TODO',
    },

    DeIdentificationModel : {
        'default': 'TODO',
    },
    RelationExtractionModel : {
        'default': 'TODO',
    },

    RelationExtractionDLModel : {
        'default': 'TODO',
    },
    Chunk2Token : {
        'default': '',# TODO
    },

    ContextualParserModel : {
        'default': '',# TODO

    },

    DrugNormalizer : {
        'default': '',# TODO
    },

    GenericClassifierModel : {
        'default': '',# TODO
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
        'default': 'TODO',
    }

}







