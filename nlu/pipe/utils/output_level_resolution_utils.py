"""Resolve output level of pipeline and components"""
from typing import List, Dict
import logging

from nlu.universe.logic_universes import NLP_LEVELS, AnnoTypes
from nlu.universe.feature_node_ids import NLP_NODE_IDS
from nlu.universe.feature_universes import NLP_FEATURES

logger = logging.getLogger('nlu')
from nlu.universe.universes import Licenses
from sparknlp.base import *
from sparknlp.annotator import *
from nlu.pipe.col_substitution.col_name_substitution_utils import ColSubstitutionUtils

"""Component and Column Level logic operations and utils"""


class OutputLevelUtils():
    levels = {
        'token': ['token', 'pos', 'ner', 'lemma', 'lem', 'stem', 'stemm', 'word_embeddings', 'named_entity',
                  'entity', 'dependency',
                  'labeled_dependency', 'dep', 'dep.untyped', 'dep.typed'],
        'sentence': ['sentence', 'sentence_embeddings', ] + ['sentiment', 'classifer', 'category'],
        'chunk': ['chunk', 'embeddings_chunk', 'chunk_embeddings'],
        'document': ['document', 'language'],
        'embedding_level': []
        # ['sentiment', 'classifer'] # WIP, wait for Spark NLP Getter/Setter fixes to implement this properly
        # embedding level  annotators output levels depend on the level of the embeddings they are fed. If we have Doc/Chunk/Word/Sentence embeddings, those annotators output at the same level.

    }
    annotator_levels_approach_based = {
        'document': [DocumentAssembler, Chunk2Doc,
                     YakeKeywordExtraction, DocumentNormalizer
                     ],
        'sentence': [SentenceDetector, SentenceDetectorDLApproach],
        'chunk': [Chunker, ChunkEmbeddings, ChunkTokenizer, Token2Chunk, TokenAssembler,
                  NerConverter, Doc2Chunk, NGramGenerator],
        'token': [NerCrfApproach, NerDLApproach,
                  PerceptronApproach,
                  Stemmer,
                  ContextSpellCheckerApproach,
                  WordSegmenterApproach,
                  Lemmatizer, LemmatizerModel, TypedDependencyParserApproach, DependencyParserApproach,
                  Tokenizer, RegexTokenizer, RecursiveTokenizer
            , DateMatcher, TextMatcher, BigTextMatcher, MultiDateMatcher,
                  WordSegmenterApproach
                  ],
        # 'sub_token': [StopWordsCleaner, DateMatcher, TextMatcher, BigTextMatcher, MultiDateMatcher],
        # these can be document or sentence
        'input_dependent': [ViveknSentimentApproach, SentimentDLApproach, ClassifierDLApproach,
                            LanguageDetectorDL,
                            MultiClassifierDLApproach, SentenceEmbeddings, NorvigSweetingApproach,
                            BertForSequenceClassification, DistilBertForTokenClassification, ],
        'multi': [MultiClassifierDLApproach, SentenceEmbeddings, NorvigSweetingApproach, ]
        # 'unclassified': [Yake, Ngram]
    }
    annotator_levels_model_based = {
        'document': [],
        'sentence': [SentenceDetectorDLModel, ],
        'chunk': [ChunkTokenizerModel, ChunkTokenizerModel, ],
        'token': [ContextSpellCheckerModel, AlbertEmbeddings, BertEmbeddings, ElmoEmbeddings, WordEmbeddings,
                  XlnetEmbeddings, WordEmbeddingsModel,
                  # NER models are token level, they give IOB predictions and cofidences for EVERY token!
                  NerDLModel, NerCrfModel, PerceptronModel, SymmetricDeleteModel, NorvigSweetingModel,
                  ContextSpellCheckerModel,
                  TypedDependencyParserModel, DependencyParserModel,
                  RecursiveTokenizerModel,
                  TextMatcherModel, BigTextMatcherModel, RegexMatcherModel,
                  WordSegmenterModel, TokenizerModel,
                  XlmRoBertaEmbeddings, RoBertaEmbeddings, DistilBertEmbeddings,
                  BertForTokenClassification, DistilBertForTokenClassification,
                  AlbertForTokenClassification, XlmRoBertaForTokenClassification,
                  RoBertaForTokenClassification, LongformerForTokenClassification,
                  XlnetForTokenClassification,
                  ],
        # 'sub_token': [TextMatcherModel, BigTextMatcherModel, RegexMatcherModel, ],
        # sub token is when annotator is token based but some tokens may be missing since dropped/cleaned
        'sub_token': [
            StopWordsCleaner, NormalizerModel

        ],
        'input_dependent': [BertSentenceEmbeddings, UniversalSentenceEncoder, ViveknSentimentModel,
                            SentimentDLModel, ClassifierDLModel,
                            MarianTransformer, T5Transformer,
                            XlmRoBertaEmbeddings, RoBertaEmbeddings, DistilBertEmbeddings,

                            ],
        'multi': [MultiClassifierDLModel, MultiClassifierDLModel, ]
    }
    all_embeddings = {
        'token': [AlbertEmbeddings, BertEmbeddings, ElmoEmbeddings, WordEmbeddings,
                  XlnetEmbeddings, WordEmbeddingsModel],
        'input_dependent': [SentenceEmbeddings, UniversalSentenceEncoder, BertSentenceEmbeddings]

    }

    @staticmethod
    def infer_output_level(pipe):
        '''
        This function checks the LAST  component of the NLU pipeline and infers
        and infers from that the output level via checking the components' info.
        It sets the output level of the component_list accordingly
        '''
        if pipe.output_level == '':
            # Loop in reverse over component_list and get first non util/sentence_detecotr/tokenizer/doc_assember. If there is non, take last
            bad_types = [AnnoTypes.HELPER_ANNO, AnnoTypes.SENTENCE_DETECTOR]
            bad_names = [NLP_NODE_IDS.TOKENIZER]
            for c in pipe.components[::-1]:
                if any(t in c.type for t in bad_types):
                    continue
                if any(n in c.name for n in bad_names):
                    continue
                pipe.output_level = OutputLevelUtils.resolve_component_to_output_level(pipe, c)
                logger.info(f'Inferred and set output level of pipeline to {pipe.output_level}', )
                break
            # Normalizer bug that does not happen in debugger bugfix
            if pipe.output_level is None or pipe.output_level == '':
                pipe.output_level = NLP_LEVELS.DOCUMENT
            logger.info(f'Inferred and set output level of pipeline to {pipe.output_level}')

        else:
            return

    @staticmethod
    def get_output_level_of_embeddings_provider(pipe, field_type, field_name):
        '''
        This function will go through all components to find the component which  generate @component_output_column_name.
        Then it will go gain through all components to find the component, from which @component_output_column_name is taking its inputs
        Then it will return the type of the provider component. This result isused to resolve the output level of the component that depends on the inpit for the output level
        :param field_type: The type of the field we want to resolve the input level for
        :param field_name: The name of the field we want to resolve the input level for

        :return:
        '''
        # find the component. Column output name should be unique
        component_inputs = []
        for component in pipe.components:
            if field_name == component.info.name:
                component_inputs = component.info.spark_input_column_names

        # get the embedding feature name
        target_output_component = ''
        for input_name in component_inputs:
            if 'embed' in input_name: target_output_component = input_name

        # get the model that outputs that feature
        for component in pipe.components:
            component_outputs = component.info.spark_output_column_names
            for input_name in component_outputs:
                if target_output_component == input_name:
                    # this is the component that feeds into the component we are trying to resolve the output  level for.
                    # That is so, because the output of this component matches the input of the component we are resolving
                    return pipe.resolve_type_to_output_level(component.info.type)

    @staticmethod
    def resolve_type_to_output_level(pipe, field_type, field_name):
        '''
        This checks the levels dict for what the output level is for the input annotator type.
        If the annotator type depends on the embedding level, we need further checking.
        @ param field_type : type of the spark field
        @ param name : name of thhe spark field
        @ return : String, which corrosponds to the output level of this Component.
        '''
        logger.info('Resolving output level for field_type=%s and field_name=%s', field_type, field_name)
        if field_name == 'sentence':
            logger.info('Resolved output level for field_type=%s and field_name=%s to Sentence level', field_type,
                        field_name)
            return 'sentence'
        if field_type in pipe.levels['token']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to Token level ', field_type,
                        field_name)
            return 'token'
        if field_type in pipe.levels['sentence']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to sentence level', field_type,
                        field_name)
            return 'sentence'
        if field_type in pipe.levels['chunk']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to Chunk level ', field_type,
                        field_name)
            return 'chunk'
        if field_type in pipe.levels['document']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to document level', field_type,
                        field_name)
            return 'document'
        if field_type in pipe.levels['embedding_level']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to embeddings level', field_type,
                        field_name)
            return pipe.get_output_level_of_embeddings_provider(field_type, field_name)  # recursive resolution

    @staticmethod
    def resolve_input_dependent_component_to_output_level(pipe, component):
        '''
        For a given NLU component  which is input dependent , resolve its output level by checking if it's input stem
        from document or sentence based annotators :param component:  to resolve :return: resolve component
        '''
        # (1.) A classifier, which is using sentence/document. We just check input cols

        if 'document' in component.spark_input_column_names:
            return 'document'
        if 'sentence' in component.spark_input_column_names:
            return 'sentence'

        # (2.) A classifier, which is using sentence/doc embeddings.
        # We iterate over the component_list and check which Embed component is feeding the classifier and what the input that embed annotator is (sent or doc)
        for c in pipe.components:
            # check if os_components is of sentence embedding class  which is always input dependent
            if any(isinstance(c.model, e) for e in OutputLevelUtils.all_embeddings['input_dependent']): # TODO refactor
                if NLP_FEATURES.DOCUMENT in c.spark_input_column_names:  return NLP_FEATURES.DOCUMENT
                if NLP_FEATURES.SENTENCE in c.spark_input_column_names:  return NLP_FEATURES.SENTENCE

    @staticmethod
    def resolve_component_to_output_level(pipe, component):
        '''
        For a given NLU component, resolve its output level, by checking annotator_levels dicts for approaches and models
        If output level is input dependent, resolve_input_dependent_component_to_output_level will resolve it
        :param component:  to resolve
        :return: resolve component
        '''
        for level in OutputLevelUtils.annotator_levels_model_based.keys():
            for t in OutputLevelUtils.annotator_levels_model_based[level]:
                if isinstance(component.model, t):
                    if level == 'input_dependent':
                        return OutputLevelUtils.resolve_input_dependent_component_to_output_level(pipe, component)
                    else:
                        return level
        for level in OutputLevelUtils.annotator_levels_approach_based.keys():
            for t in OutputLevelUtils.annotator_levels_approach_based[level]:
                if isinstance(component.model, t):
                    if level == 'input_dependent':
                        return OutputLevelUtils.resolve_input_dependent_component_to_output_level(pipe, component)
                    else:
                        return level

        if pipe.has_licensed_components:
            from nlu.pipe.extractors.output_level_HC_map import HC_anno2output_level
            for level in HC_anno2output_level.keys():
                for t in HC_anno2output_level[level]:
                    if isinstance(component.model, t):
                        if level == 'input_dependent':
                            return OutputLevelUtils.resolve_input_dependent_component_to_output_level(pipe, component)
                        else:
                            return level

    @staticmethod
    def get_output_level_mappings(pipe, df, anno_2_ex_config, get_embeddings):
        """Get a dict where key=spark_colname and val=output_level, inferred from processed dataframe and
        component_list that is currently running """
        output_level_map = {}
        same_output_level_map = {}
        not_same_output_level_map = {}
        for c in pipe.components:
            if 'embedding' in c.type and get_embeddings == False: continue
            generated_cols = ColSubstitutionUtils.get_final_output_cols_of_component(c, df, anno_2_ex_config)
            output_level = OutputLevelUtils.resolve_component_to_output_level(pipe, c)
            if output_level == pipe.output_level:
                for g_c in generated_cols: same_output_level_map[g_c] = output_level
            else:
                for g_c in generated_cols: not_same_output_level_map[g_c] = output_level
            for g_c in generated_cols: output_level_map[g_c] = output_level
        return output_level_map, same_output_level_map, not_same_output_level_map

    @staticmethod
    def get_cols_at_same_output_level(pipe, df, anno_2_ex_config, col2output_level: Dict[str, str]) -> List[str]:
        """Get List of cols which are at same output level as the component_list is currently configured to"""
        same_output_level_cols = []
        for c in pipe.components:
            if col2output_level[c.out_types[0]] == pipe.output_level:
                same_output_level_cols + ColSubstitutionUtils.get_final_output_cols_of_component(c, df,
                                                                                                 anno_2_ex_config)
        return same_output_level_cols

    @staticmethod
    def get_cols_not_at_same_output_level(pipe, df, anno_2_ex_config, col2output_level: Dict[str, str]) -> List[str]:
        """Get List of cols which are not at same output level as the component_list is currently configured to"""
        return [c.out_types[0] for c in pipe.components if
                not col2output_level[c.out_types[0]] == pipe.output_level]

    @staticmethod
    def get_output_level_mapping_by_component(pipe) -> Dict[str, str]:
        """Get a dict where key=colname and val=output_level, inferred from processed dataframe and component_list
        that is currently running """
        nlp_levels = {c: OutputLevelUtils.resolve_component_to_output_level(pipe, c) for c in pipe.components}
        for c in pipe.components :
            if c.license == Licenses.ocr:
                nlp_levels[c] = c.output_level
        return {c: OutputLevelUtils.resolve_component_to_output_level(pipe, c) for c in pipe.components}
