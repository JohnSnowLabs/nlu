from nlu.pipe.extractors.extractor_methods.base_extractor_methods import *
from nlu.pipe.extractors.extractor_methods.helper_extractor_methods import *

"""
This file contains methods to get pre-defined configurations for every annotator.
Extractor_resolver.py should be used to resolve SparkNLP Annotator classes to methods 
in this file, which return the corrosponding configs that need to be passed to 
the master_extractor() call.

This file is where all the in extractor_base_data_classes.py Dataclasses are combined with the 
extractors defined in extractor_methods.py.

"""


def default_get_nothing(output_col_prefix):
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        name='nothing_extractor',
        description='Extracts nothing. Useful for annotators with irrelevant data'
    )


def default_only_result_config(output_col_prefix):
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=True,
        name='Default result extractor',
        description='Just gets the result field'
    )


def default_full_config(output_col_prefix='DEFAULT'):
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_positions=True,
        get_begin=True,
        get_end=True,
        get_embeds=True,
        get_result=True,
        get_meta=True,
        get_full_meta=True,
        get_annotator_type=True,
        name='default_full',
        description='Default full configuration, keeps all data and gets all metadata fields',

    )


def default_NER_converter_licensed_config(output_col_prefix='entities'):
    """Extracts NER tokens withouth positions, just the converted IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=True,
        get_meta=True,
        meta_white_list=['entity', 'confidence'],  # sentence, chunk
        name='default_ner',
        description='Converts IOB-NER representation into entity representation and generates confidences for the entire entity chunk',
    )



def default_chunk_mapper_config(output_col_prefix='mapped_entity'):
    """Extracts NER tokens withouth positions, just the converted IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=True,
        get_meta=True,
        meta_white_list=['relation', 'all_relations','chunk', 'entity',
                         'sentence'
                         ],  # MAYBE DROP 'chunk', 'entity'default_chunk_mapper_config, sentence
        name='default_ner',
        meta_data_extractor=SparkNLPExtractor(extract_chunk_mapper_relation_data,
                                          'Get ChunkMapper Relation Metadata',
                                          'Get ChunkMapper Relation Metadata'),
        description='Extract Chunk Mapper with relation Data',
    )



def default_chunk_resolution_config(output_col_prefix='resolved_entities'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=True,
        get_meta=True,
        meta_white_list=['confidence', 'resolved_text'],  # sentence, chunk
        name='default_ner',

    )


def full_resolver_config(output_col_prefix='DEFAULT'):
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_positions=True,
        get_begin=True,
        get_end=True,
        get_embeds=True,
        get_result=True,
        get_meta=True,
        get_full_meta=True,
        get_annotator_type=True,
        name='default_full',
        description='Full resolver outputs, with any _k_ field in the metadata dict splitted :::',
        meta_data_extractor=SparkNLPExtractor(extract_resolver_all_k_subfields_splitted,
                                              'Splits all _k_ fields on :::d returns all other fields as corrosponding to pop config',
                                              'split all _k_ fields')

    )


def resolver_conifg_with_metadata(output_col_prefix='DEFAULT'):
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_meta=True,
        get_result=True,
        get_full_meta=True,
        name='with metadata',
        description='Full resolver outputs, with any _k_ field in the metadata dict splitted :::',
        meta_data_extractor=SparkNLPExtractor(extract_resolver_all_k_subfields_splitted,
                                              'Splits all _k_ fields on :::d returns all other fields as corrosponding to pop config',
                                              'split all _k_ fields')

    )


def default_relation_extraction_positional_config(output_col_prefix='extracted_relations'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=True,
        get_meta=True,
        get_full_meta=True,
        name='full_relation_extraction',
        description='Get relation extraction result and all metadata, with positions of entities',
    )


def default_relation_extraction_config(output_col_prefix='extracted_relations'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=True,
        meta_white_list=[],
        get_meta=True,
        meta_black_list=['entity1_begin', 'entity2_begin', 'entity1_end', 'entity2_end', ],
        name='default_relation_extraction',
        description='Get relation extraction result and all metadata, positions of entities excluded',
    )


def default_de_identification_config(output_col_prefix='de_identified'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=True,
        name='positional_relation_extraction',
        description='Get relation extraction result and all metadata, which will include positions of entities chunks',
    )


def default_assertion_config(output_col_prefix='assertion'):
    """Extracts NER tokens withouth positions, just the IOB tags,confidences and classified tokens """
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=True,
        name='default_assertion_extraction',
        get_meta=True,
        meta_white_list=['confidence'],
        description='Gets the assertion result and confidence',
    )


def default_ner_config(output_col_prefix='med_ner'):
    return default_only_result_config(output_col_prefix)


def default_ner_config(output_col_prefix='med_ner'):
    return default_get_nothing(output_col_prefix)


def default_feature_assembler_config(output_col_prefix='feature_assembler'):
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=False,
        name='features_assembled',
        get_meta=False,
        description='Gets nothing',
    )


def default_generic_classifier_config(output_col_prefix='generic_classifier'):
    return SparkNLPExtractorConfig(
        output_col_prefix=output_col_prefix,
        get_result=True,
        name='generic_classifier',
        get_meta=True,
        meta_white_list=['confidence'],
        description='Gets the  result and confidence',
    )
