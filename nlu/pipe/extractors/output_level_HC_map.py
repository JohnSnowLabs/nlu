"""
Map Healthcare annotators to output level
"""
from sparknlp_jsl.annotator  import *
HC_anno2output_level = {
    'document': [
        RENerChunksFilter,
        DrugNormalizer,

                 ],
    'sentence': [
        SentenceEntityResolverModel

    ],
    'chunk': [
        NerOverwriter,
        NerDisambiguatorModel,
        ChunkMergeModel,
        NerConverterInternal,
        ChunkEntityResolverModel,
        DeIdentificationModel,
        AssertionDLModel,
        AssertionLogRegModel,
        ContextualParserApproach,
        ContextualParserModel


    ],
    'token': [
        MedicalNerModel,

               ],
    'input_dependent': [
        GenericClassifierModel,
                        ],
    'sub_token': [DrugNormalizer,Chunk2Token],
    'leveless': [
        ContextualParserModel,
        ContextualParserModel,

        ContextualParserModel,

    ],
    'relation':[
        RelationExtractionModel,
        PosologyREModel,
        RelationExtractionDLModel,
    ]

}

