# NLU Healthcare components

# Main components

# 0 Base internal Spark NLP structure.md required for all JSL components

# we cant call the embdding file "embeddings" because namespacing wont let us import the Embeddings class inside of it then

# sentence
# Embeddings

# classifiers

# matchers

# token level operators

## spell

# sentence

# seq2seq


from nlu.components.assertions.assertion_dl.assertion_dl import AssertionDL
from nlu.components.assertions.assertion_log_reg.assertion_log_reg import AssertionLogReg
from nlu.components.chunkers.contextual_parser.contextual_parser import ContextualParser
from nlu.components.classifiers.generic_classifier.generic_classifier import GenericClassifier
from nlu.components.classifiers.ner_healthcare.ner_dl_healthcare import NERDLHealthcare
from nlu.universe.annotator_class_universe import AnnoClassRef
from nlu.universe.universes import ComponentBackends, ComputeContexts
from nlu.universe.logic_universes import NLP_LEVELS, NLP_ANNO_TYPES
from nlu.components.deidentifiers.deidentifier.deidentifier import Deidentifier
from nlu.components.normalizers.drug_normalizer.drug_normalizer import DrugNorm
from nlu.components.relation_extractors.relation_extractor.relation_extractor import RelationExtraction
from nlu.components.resolutions.sentence_entity_resolver.sentence_resolver import SentenceResolver
from nlu.components.utils.ner_to_chunk_converter_licensed.ner_to_chunk_converter_licensed import \
    NerToChunkConverterLicensed
from nlu.pipe.col_substitution.col_substitution_HC import substitute_assertion_cols, substitute_context_parser_cols, \
    substitute_de_identification_cols, substitute_drug_normalizer_cols, substitute_generic_classifier_parser_cols, \
    substitute_ner_internal_converter_cols, substitute_relation_cols, substitute_sentence_resolution_cols
from nlu.pipe.col_substitution.col_substitution_OS import substitute_ner_dl_cols
from nlu.pipe.extractors.extractor_configs_HC import default_assertion_config, default_full_config, \
    default_de_identification_config, default_only_result_config, default_generic_classifier_config, default_ner_config, \
    default_NER_converter_licensed_config, default_relation_extraction_config, \
    default_relation_extraction_positional_config, default_chunk_resolution_config
from nlu.universe.feature_node_ids import NLP_NODE_IDS, NLP_HC_NODE_IDS
from nlu.universe.feature_node_universes import NLP_HC_FEATURE_NODES
from nlu.universe.feature_universes import NLP_FEATURES, OCR_FEATURE, NLP_HC_FEATURES
from nlu.pipe.nlu_component import NluComponent
from nlu.universe.universes import Licenses, ComputeContexts

class ComponentMapHC:
    # Encapsulate all Healthcare components Constructors by mappping each individual Annotator class to a specific Construction
    A = NLP_NODE_IDS
    H_A = NLP_HC_NODE_IDS
    T = NLP_ANNO_TYPES
    F = NLP_FEATURES
    L = NLP_LEVELS
    ACR = AnnoClassRef
    hc_components = {
        # TODO THIS SHOULD BE A SEPERATED CLASS which ONLY INSTATIATE when LICENSE VALIDATE!!!>
        H_A.ASSERTION_DL: NluComponent(
            name=H_A.ASSERTION_DL,
            type=T.CHUNK_CLASSIFIER,
            get_default_model=AssertionDL.get_default_model,
            get_pretrained_model=AssertionDL.get_pretrained_model,
            get_trainable_model=AssertionDL.get_default_trainable_model,
            pdf_extractor_methods={'default': default_assertion_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_assertion_cols,
            output_level=L.NER_CHUNK,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.ASSERTION_DL],
            description='Deep Learning based Assertion model that maps NER-Chunks into a pre-defined terminology.',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.ASSERTION_DL,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.ASSERTION_DL],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.ASSERTION_DL],
            has_storage_ref=True,
            is_storage_ref_consumer=True,
            trainable_mirror_anno=H_A.TRAINABLE_ASSERTION_DL
        ),
        H_A.TRAINABLE_ASSERTION_DL: NluComponent(
            name=H_A.TRAINABLE_ASSERTION_DL,
            type=T.CHUNK_CLASSIFIER,
            get_default_model=AssertionDL.get_default_model,
            get_pretrained_model=AssertionDL.get_pretrained_model,
            get_trainable_model=AssertionDL.get_default_trainable_model,
            pdf_extractor_methods={'default': default_assertion_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_assertion_cols,
            output_level=L.NER_CHUNK,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.TRAINABLE_ASSERTION_DL],
            description='Trainable Deep Learning based Assertion model that maps NER-Chunks into a pre-defined terminology.',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.TRAINABLE_ASSERTION_DL,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.TRAINABLE_ASSERTION_DL],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.TRAINABLE_ASSERTION_DL],
            has_storage_ref=True,
            is_storage_ref_consumer=True,
            trainable=True,
            trained_mirror_anno=H_A.ASSERTION_DL),
        # H_A.ASSERTION_FILTERER: NluComponent( # TODO not integrated
        #     name=H_A.ASSERTION_FILTERER,
        #     type=T.CHUNK_FILTERER,
        #     get_default_model=AssertionDL.get_default_model,
        #     get_pretrained_model=AssertionDL.get_pretrained_model,
        #     get_trainable_model=AssertionDL.get_default_trainable_model,
        #     pdf_extractor_methods={'default': default_assertion_config, 'default_full': default_full_config, },
        #     pdf_col_name_substitutor=substitute_assertion_cols,
        #     output_level=L.NER_CHUNK,
        #     node=NLP_HC_FEATURE_NODES.ASSERTION_DL,
        #     description='Trainable Deep Learning based Assertion model that maps NER-Chunks into a pre-defined terminology.',
        #     provider=ComponentBackends.hc,
        #     license=Licenses.hc,
        #     computation_context=ComputeContexts.spark,
        #     output_context=ComputeContexts.spark,
        #     jsl_anno_class=H_A.ASSERTION_FILTERER,
        #     jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.ASSERTION_FILTERER],
        #     jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.ASSERTION_FILTERER],
        #     has_storage_ref=True,
        #     is_is_storage_ref_consumer=True,
        #     trainable=True,
        #     trained_mirror_anno=H_A.ASSERTION_FILTERER), AssertionLogReg
        H_A.ASSERTION_LOG_REG: NluComponent(
            name=H_A.ASSERTION_LOG_REG,
            type=T.CHUNK_CLASSIFIER,
            get_default_model=AssertionLogReg.get_default_model,
            get_pretrained_model=AssertionLogReg.get_pretrained_model,
            get_trainable_model=AssertionLogReg.get_default_trainable_model,
            pdf_extractor_methods={'default': default_assertion_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_assertion_cols,
            output_level=L.NER_CHUNK,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.ASSERTION_LOG_REG],
            description='Classical ML based Assertion model that maps NER-Chunks into a pre-defined terminology.',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.ASSERTION_LOG_REG,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.ASSERTION_LOG_REG],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.ASSERTION_LOG_REG],
            trained_mirror_anno=H_A.TRAINABLE_ASSERTION_LOG_REG),
        H_A.TRAINABLE_ASSERTION_LOG_REG: NluComponent(
            name=H_A.TRAINABLE_ASSERTION_LOG_REG,
            type=T.CHUNK_CLASSIFIER,
            get_default_model=AssertionLogReg.get_default_model,
            get_pretrained_model=AssertionLogReg.get_pretrained_model,
            get_trainable_model=AssertionLogReg.get_default_trainable_model,
            pdf_extractor_methods={'default': default_assertion_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_assertion_cols,
            output_level=L.NER_CHUNK,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.TRAINABLE_ASSERTION_LOG_REG],
            description='Classical ML based Assertion model that maps NER-Chunks into a pre-defined terminology.',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.TRAINABLE_ASSERTION_LOG_REG,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.TRAINABLE_ASSERTION_LOG_REG],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.TRAINABLE_ASSERTION_LOG_REG],
            trained_mirror_anno=H_A.ASSERTION_LOG_REG),
        H_A.CHUNK2TOKEN: 'TODO not integrated',
        H_A.CHUNK_ENTITY_RESOLVER: 'Deprecated',
        H_A.TRAINABLE_CHUNK_ENTITY_RESOLVER: 'Deprecated',
        H_A.CHUNK_FILTERER: 'TODO not integrated',
        H_A.CHUNK_KEY_PHRASE_EXTRACTION: 'TODO not integrated',
        H_A.CHUNK_MERGE: 'TODO not integrated',
        H_A.CONTEXTUAL_PARSER: NluComponent(
            name=H_A.CONTEXTUAL_PARSER,
            type=T.CHUNK_CLASSIFIER,
            get_default_model=ContextualParser.get_default_model,
            get_trainable_model=ContextualParser.get_trainable_model,
            pdf_extractor_methods={'default': default_full_config, 'default_full': default_full_config, },
            # TODO extractr method
            pdf_col_name_substitutor=substitute_context_parser_cols,
            output_level=L.NER_CHUNK,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.CONTEXTUAL_PARSER],
            description='Rule based entity extractor.',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.CONTEXTUAL_PARSER,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.CONTEXTUAL_PARSER],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.CONTEXTUAL_PARSER]),
        H_A.DE_IDENTIFICATION: NluComponent(
            name=H_A.DE_IDENTIFICATION,
            type=T.CHUNK_CLASSIFIER,
            get_default_model=Deidentifier.get_default_model,
            get_pretrained_model=Deidentifier.get_pretrained_model,
            pdf_extractor_methods={'default': default_de_identification_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_de_identification_cols,
            output_level=L.DOCUMENT,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.DE_IDENTIFICATION],
            description='De-Identify named entity according to various Healthcare Data Protection standards',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.DE_IDENTIFICATION,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.DE_IDENTIFICATION],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.DE_IDENTIFICATION]),
        H_A.DOCUMENT_LOG_REG_CLASSIFIER: 'TODO not integrated',
        H_A.TRAINABLE_DOCUMENT_LOG_REG_CLASSIFIER: 'TODO not integrated',
        H_A.DRUG_NORMALIZER: NluComponent(
            name=H_A.DRUG_NORMALIZER,
            type=T.CHUNK_CLASSIFIER,
            get_default_model=DrugNorm.get_default_model,
            pdf_extractor_methods={'default': default_only_result_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_drug_normalizer_cols,
            output_level=L.DOCUMENT,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.DRUG_NORMALIZER],
            description='Normalizes raw clinical and crawled text which contains drug names into cleaned and standardized representation',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.DRUG_NORMALIZER,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.DRUG_NORMALIZER],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.DRUG_NORMALIZER]),  #
        # H_A.FEATURES_ASSEMBLER: NluComponent( # TODO partially integrated. featire mpde ,ossomg
        #     name=H_A.FEATURES_ASSEMBLER,
        #     type=T.HELPER_ANNO,
        #     get_default_model=SparkNLPFeatureAssembler.get_default_model,
        #     pdf_extractor_methods={'default': default_feature_assembler_config, 'default_full': default_full_config, },
        #     # pdf_col_name_substitutor=substitute_drug_normalizer_cols, # TODO no substition
        #     output_level=L.DOCUMENT, # TODO double check output level?
        #     node=NLP_HC_FEATURE_NODES.FEATURES_ASSEMBLER,
        #     description='Aggregated features from various annotators into one column for training generic classifiers',
        #     provider=ComponentBackends.hc,
        #     license=Licenses.hc,
        #     computation_context=ComputeContexts.spark,
        #     output_context=ComputeContexts.spark,
        #     jsl_anno_class=H_A.FEATURES_ASSEMBLER,
        #     jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.FEATURES_ASSEMBLER],
        #     jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.FEATURES_ASSEMBLER]),
        H_A.GENERIC_CLASSIFIER: NluComponent(
            name=H_A.GENERIC_CLASSIFIER,
            type=T.DOCUMENT_CLASSIFIER,
            get_default_model=GenericClassifier.get_default_model,
            get_trainable_model=GenericClassifier.get_default_model,
            get_pretrained_model=GenericClassifier.get_default_model,
            pdf_extractor_methods={'default': default_generic_classifier_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_generic_classifier_parser_cols,
            output_level=L.DOCUMENT,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.GENERIC_CLASSIFIER],
            description='Generic Deep Learning based tensorflow classifier',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.GENERIC_CLASSIFIER,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.GENERIC_CLASSIFIER],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.GENERIC_CLASSIFIER],
            trainable_mirror_anno=H_A.TRAINABLE_GENERIC_CLASSIFIER
        ),
        H_A.TRAINABLE_GENERIC_CLASSIFIER: NluComponent(
            name=H_A.TRAINABLE_GENERIC_CLASSIFIER,
            type=T.DOCUMENT_CLASSIFIER,
            get_default_model=GenericClassifier.get_default_model,
            get_trainable_model=GenericClassifier.get_default_model,
            get_pretrained_model=GenericClassifier.get_default_model,
            pdf_extractor_methods={'default': default_generic_classifier_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_generic_classifier_parser_cols,
            output_level=L.DOCUMENT,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.GENERIC_CLASSIFIER],
            description='Generic Deep Learning based tensorflow classifier',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.TRAINABLE_GENERIC_CLASSIFIER,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.TRAINABLE_GENERIC_CLASSIFIER],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.TRAINABLE_GENERIC_CLASSIFIER],
            trained_mirror_anno=H_A.GENERIC_CLASSIFIER
        ),
        H_A.IOB_TAGGER: 'TODO not integrated',
        H_A.MEDICAL_NER: NluComponent(
            name=H_A.MEDICAL_NER,
            type=T.CHUNK_CLASSIFIER,
            get_default_model=NERDLHealthcare.get_default_model,
            get_trainable_model=NERDLHealthcare.get_default_model,
            get_pretrained_model=NERDLHealthcare.get_default_model,
            pdf_extractor_methods={'default': default_ner_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_ner_dl_cols,
            output_level=L.NER_CHUNK,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.MEDICAL_NER],
            description='Deep Learning based Medical Named Entity Recognizer (NER)',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.MEDICAL_NER,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.MEDICAL_NER],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.MEDICAL_NER],
            trainable_mirror_anno=H_A.TRAINABLE_MEDICAL_NER,
            has_storage_ref=True,
            is_storage_ref_consumer=True
        ),
        H_A.TRAINABLE_MEDICAL_NER: NluComponent(
            name=H_A.TRAINABLE_MEDICAL_NER,
            type=T.CHUNK_CLASSIFIER,
            get_default_model=NERDLHealthcare.get_default_model,
            get_trainable_model=NERDLHealthcare.get_default_model,
            get_pretrained_model=NERDLHealthcare.get_default_model,
            pdf_extractor_methods={'default': default_ner_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_ner_dl_cols,
            output_level=L.NER_CHUNK,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.TRAINABLE_MEDICAL_NER],
            description='Trainable Deep Learning based Medical Named Entity Recognizer (NER)',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.TRAINABLE_MEDICAL_NER,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.TRAINABLE_MEDICAL_NER],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.TRAINABLE_MEDICAL_NER],
            trained_mirror_anno=H_A.TRAINABLE_MEDICAL_NER,
            has_storage_ref=True,
            is_storage_ref_consumer=True
        ),
        H_A.NER_CHUNKER: 'TODO not integrated',
        H_A.NER_CONVERTER_INTERNAL: NluComponent(
            name=H_A.NER_CONVERTER_INTERNAL,
            type=T.HELPER_ANNO,
            get_default_model=NerToChunkConverterLicensed.get_default_model,
            pdf_extractor_methods={'default': default_NER_converter_licensed_config,
                                   'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_ner_internal_converter_cols,
            output_level=L.NER_CHUNK,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.NER_CONVERTER_INTERNAL],
            description='Convert NER-IOB tokens into concatenated strings (aka chunks)',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.NER_CONVERTER_INTERNAL,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.NER_CONVERTER_INTERNAL],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.NER_CONVERTER_INTERNAL],
        ),
        H_A.NER_DISAMBIGUATOR: 'TODO not integrated',
        H_A.RELATION_NER_CHUNKS_FILTERER: 'TODO not integrated',
        H_A.RE_IDENTIFICATION: 'TODO not integrated',
        H_A.RELATION_EXTRACTION: NluComponent(
            name=H_A.RELATION_EXTRACTION,
            type=T.RELATION_CLASSIFIER,
            get_default_model=RelationExtraction.get_default_model,
            get_pretrained_model=RelationExtraction.get_pretrained_model,
            get_trainable_model=RelationExtraction.get_default_trainable_model,
            pdf_extractor_methods={'default': default_relation_extraction_config,
                                   'positional': default_relation_extraction_positional_config,
                                   'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_relation_cols,
            output_level=L.RELATION,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.RELATION_EXTRACTION],
            description='Classical ML model for predicting relation ship between entity pairs',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.RELATION_EXTRACTION,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.RELATION_EXTRACTION],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.RELATION_EXTRACTION],
            trainable_mirror_anno=H_A.TRAINABLE_RELATION_EXTRACTION
        ),
        H_A.TRAINABLE_RELATION_EXTRACTION: NluComponent(
            name=H_A.TRAINABLE_RELATION_EXTRACTION,
            type=T.RELATION_CLASSIFIER,
            get_default_model=RelationExtraction.get_default_model,
            get_pretrained_model=RelationExtraction.get_pretrained_model,
            get_trainable_model=RelationExtraction.get_default_trainable_model,
            pdf_extractor_methods={'default': default_relation_extraction_config,
                                   'positional': default_relation_extraction_positional_config,
                                   'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_relation_cols,
            output_level=L.RELATION,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.TRAINABLE_RELATION_EXTRACTION],
            description='Trainable Classical ML model for predicting relation ship between entity pairs',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.TRAINABLE_RELATION_EXTRACTION,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.TRAINABLE_RELATION_EXTRACTION],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.TRAINABLE_RELATION_EXTRACTION],
            trained_mirror_anno=H_A.RELATION_EXTRACTION,
            trainable=True
        ),
        H_A.RELATION_EXTRACTION_DL: NluComponent(
            name=H_A.RELATION_EXTRACTION_DL,
            type=T.RELATION_CLASSIFIER,
            get_default_model=RelationExtraction.get_default_model,
            get_pretrained_model=RelationExtraction.get_pretrained_model,
            get_trainable_model=RelationExtraction.get_default_trainable_model,
            pdf_extractor_methods={'default': default_relation_extraction_config,
                                   'positional': default_relation_extraction_positional_config,
                                   'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_relation_cols,
            output_level=L.RELATION,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.TRAINABLE_RELATION_EXTRACTION],
            description='Deep Learning based model for predicting relation ship between entity pairs',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.RELATION_EXTRACTION_DL,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.RELATION_EXTRACTION_DL],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.RELATION_EXTRACTION_DL],
            # trainable_mirror_anno=H_A.TRAINABLE_RELATION_EXTRACTION_DL
        ),
        # H_A.TRAINABLE_RELATION_EXTRACTION_DL: NluComponent( # DOES NOT EXIST!
        #     name=H_A.TRAINABLE_RELATION_EXTRACTION_DL,
        #     type=T.RELATION_CLASSIFIER,
        #     get_default_model=RelationExtractionDL.get_default_model,
        #     get_pretrained_model=RelationExtractionDL.get_pretrained_model,
        #     pdf_extractor_methods={ 'default': default_relation_extraction_config, 'positional': default_relation_extraction_positional_config, 'default_full'  : default_full_config, },
        #     pdf_col_name_substitutor=substitute_relation_cols,
        #     output_level=L.RELATION,
        #     node=NLP_HC_FEATURE_NODES.TRAINABLE_RELATION_EXTRACTION_DL,
        #     description='Trainable Deep Learning based model for predicting relation ship between entity pairs',
        #     provider=ComponentBackends.hc,
        #     license=Licenses.hc,
        #     computation_context=ComputeContexts.spark,
        #     output_context=ComputeContexts.spark,
        #     jsl_anno_class=H_A.TRAINABLE_RELATION_EXTRACTION_DL,
        #     jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.TRAINABLE_RELATION_EXTRACTION_DL],
        #     jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.TRAINABLE_RELATION_EXTRACTION_DL],
        #     trained_mirror_anno=H_A.RELATION_EXTRACTION_DL,
        #     trainable=True
        # ),
        H_A.SENTENCE_ENTITY_RESOLVER: NluComponent(
            name=H_A.SENTENCE_ENTITY_RESOLVER,
            type=T.CHUNK_CLASSIFIER,
            get_pretrained_model=SentenceResolver.get_pretrained_model,
            get_trainable_model=SentenceResolver.get_default_trainable_model,
            pdf_extractor_methods={'default': default_chunk_resolution_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_sentence_resolution_cols,
            output_level=L.RELATION,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.SENTENCE_ENTITY_RESOLVER],
            description='Deep Learning based entity resolver which extracts resolved entities directly from Sentence Embedding. No NER model required.',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.SENTENCE_ENTITY_RESOLVER,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.SENTENCE_ENTITY_RESOLVER],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.SENTENCE_ENTITY_RESOLVER],
            trained_mirror_anno=H_A.TRAINABLE_SENTENCE_ENTITY_RESOLVER,
            is_storage_ref_consumer=True,
            has_storage_ref=True
        ),
        H_A.TRAINABLE_SENTENCE_ENTITY_RESOLVER: NluComponent(
            name=H_A.TRAINABLE_SENTENCE_ENTITY_RESOLVER,
            type=T.CHUNK_CLASSIFIER,
            get_pretrained_model=SentenceResolver.get_pretrained_model,
            get_trainable_model=SentenceResolver.get_default_trainable_model,
            pdf_extractor_methods={'default': default_chunk_resolution_config, 'default_full': default_full_config, },
            pdf_col_name_substitutor=substitute_sentence_resolution_cols,
            output_level=L.RELATION,
            node=NLP_HC_FEATURE_NODES.nodes[H_A.TRAINABLE_SENTENCE_ENTITY_RESOLVER],
            description='Trainable Deep Learning based entity resolver which extracts resolved entities directly from Sentence Embedding. No NER model required.',
            provider=ComponentBackends.hc,
            license=Licenses.hc,
            computation_context=ComputeContexts.spark,
            output_context=ComputeContexts.spark,
            jsl_anno_class=H_A.TRAINABLE_SENTENCE_ENTITY_RESOLVER,
            jsl_anno_py_class=ACR.JSL_anno_HC_ref_2_py_class[H_A.TRAINABLE_SENTENCE_ENTITY_RESOLVER],
            jsl_anno_java_class=ACR.JSL_anno_HC_ref_2_java_class[H_A.TRAINABLE_SENTENCE_ENTITY_RESOLVER],
            trained_mirror_anno=H_A.TRAINABLE_SENTENCE_ENTITY_RESOLVER,
            is_storage_ref_consumer=True,
            has_storage_ref=True
        ),
    }