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

    ],
    'chunk': [
        NerOverwriter,
        NerDisambiguatorModel,
        ChunkMergeModel,
        NerConverterInternal,
        # ChunkEntityResolverModel, # Deprecated?
        DeIdentificationModel,
        AssertionDLModel,
        AssertionLogRegModel,
        ContextualParserApproach,
        ContextualParserModel,
        SentenceEntityResolverModel,
        SentenceEntityResolverApproach


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

