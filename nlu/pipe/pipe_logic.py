import logging

from nlu import Licenses
from nlu.pipe.nlu_component import NluComponent
from nlu.universe.component_universes import jsl_id_to_empty_component
from nlu.universe.feature_node_ids import NLP_NODE_IDS, NLP_HC_NODE_IDS
from nlu.universe.feature_universes import NLP_FEATURES
from nlu.universe.logic_universes import AnnoTypes

logger = logging.getLogger('nlu')
from nlu.pipe.utils.pipe_utils import PipeUtils
from nlu.pipe.utils.component_utils import ComponentUtils
from nlu.pipe.utils.resolution.storage_ref_utils import StorageRefUtils
from dataclasses import dataclass
from nlu.pipe.component_resolution import resolve_feature


@dataclass
class StorageRefConversionResolutionData:
    """Hold information that can be used to resolve to a NLU component_to_resolve, which satisfies the storage ref demands."""
    storage_ref: str  # storage ref a resolver component_to_resolve should have
    component_candidate: NluComponent  # from which NLU component_to_resolve should the converter feed
    type: str  # what kind of conversion, either word2chunk or word2sentence


class PipelineQueryVerifier:
    '''
        Pass a list of NLU components to the pipeline (or a NLU pipeline)
        For every component_to_resolve, it checks if all requirements are met.
        It checks and fixes the following issues  for a list of components:
        1. Missing Features / component_to_resolve requirements
        2. Bad order of components (which will cause missing features exception)
        3. Check Feature names in the output
        4. Check weather pipeline needs to be fitted
    '''

    @staticmethod
    def check_if_storage_ref_is_satisfied_or_get_conversion_candidate(component_to_check: NluComponent, pipe,
                                                                      storage_ref_to_find: str):
        """Check if any other component_to_resolve in the pipeline has same storage ref as the input component_to_resolve.
        Returns 1. If there is a candidate, but it has different level, it will be returned as candidate
        If first condition is not satisfied, consults the namespace.storage_ref_2_nlp_ref
        """
        # If there is just 1 component_to_resolve, there is nothing to check
        if len(pipe.components) == 1:
            return False, None
        conversion_candidate = None
        conversion_type = "no_conversion"
        logger.info(f'checking for storage={storage_ref_to_find} is available in component_list..')
        for c in pipe.components:
            if component_to_check.name != c.name:
                if StorageRefUtils.has_storage_ref(c):
                    if StorageRefUtils.extract_storage_ref(c) == storage_ref_to_find:
                        # Both components have Different Names AND their Storage Ref Matches up AND they both take in tokens -> Match
                        if NLP_FEATURES.TOKEN in component_to_check.in_types and c.type == AnnoTypes.TOKEN_EMBEDDING:
                            logger.info(f'Word Embedding Match found = {c.name}')
                            return False, None

                        # Since document and be substituted for sentence
                        # and vice versa if either of them matches up we have a match
                        if NLP_FEATURES.SENTENCE_EMBEDDINGS in component_to_check.in_types and \
                                c.type == AnnoTypes.DOCUMENT_EMBEDDING:
                            logger.info(f'Sentence Embedding Match found = {c.name}')
                            return False, None

                        # component_to_check requires Sentence_embedding
                        # but the Matching Storage_ref component_to_resolve takes in Token
                        #   -> Convert the Output of the Match to SentenceLevel
                        #   and feed the component_to_check to the new component_to_resolve
                        if NLP_FEATURES.SENTENCE_EMBEDDINGS in component_to_check.in_types \
                                and c.type == AnnoTypes.TOKEN_EMBEDDING:
                            logger.info(f'Sentence Embedding Conversion Candidate found={c.name}')
                            conversion_type = 'word2sentence'
                            conversion_candidate = c

                        # analogous case as above for chunk
                        if NLP_FEATURES.CHUNK_EMBEDDINGS in component_to_check.in_types and c.type == AnnoTypes.TOKEN_EMBEDDING:
                            logger.info(f'Sentence Embedding Conversion Candidate found={c.name}')
                            conversion_type = 'word2chunk'
                            conversion_candidate = c

        logger.info(f'No matching storage ref found')
        return True, StorageRefConversionResolutionData(storage_ref_to_find, conversion_candidate, conversion_type)

    @staticmethod
    def extract_required_features_refless_from_pipe(pipe):
        """Extract provided features from component_list, which have no storage ref"""
        provided_features_no_ref = []
        for c in pipe.components:
            if c.loaded_from_pretrained_pipe:
                continue
            for feat in c.in_types:
                if 'embed' not in feat: provided_features_no_ref.append(feat)
        return ComponentUtils.clean_irrelevant_features(provided_features_no_ref)

    @staticmethod
    def extract_provided_features_refless_from_pipe(pipe):
        """Extract provided features from component_list, which have no storage ref"""
        provided_features_no_ref = []
        for c in pipe.components:
            for feat in c.out_types:
                if 'embed' not in feat: provided_features_no_ref.append(feat)
        return ComponentUtils.clean_irrelevant_features(provided_features_no_ref)

    @staticmethod
    def extract_provided_features_ref_from_pipe(pipe):
        """Extract provided features from component_list, which have  storage ref.
        """
        provided_features_ref = []
        for c in pipe.components:
            for feat in c.out_types:
                if 'embed' in feat:
                    if '@' not in feat:
                        provided_features_ref.append(feat + "@" + StorageRefUtils.extract_storage_ref(c))
                    else:
                        provided_features_ref.append(feat)
        return ComponentUtils.clean_irrelevant_features(provided_features_ref)

    @staticmethod
    def extract_required_features_ref_from_pipe(pipe):
        """Extract provided features from component_list, which have  storage ref"""
        provided_features_ref = []
        for c in pipe.components:
            if c.loaded_from_pretrained_pipe:
                continue

            for feat in c.in_types:
                if 'embed' in feat:
                    # if StorageRefUtils.extract_storage_ref(os_components) !='':  # special edge case, some components might not have a storage ref set
                    if '@' not in feat:
                        provided_features_ref.append(feat + "@" + StorageRefUtils.extract_storage_ref(c))
                    else:
                        provided_features_ref.append(feat)

        return ComponentUtils.clean_irrelevant_features(provided_features_ref)

    @staticmethod
    def extract_sentence_embedding_conversion_candidates(pipe):
        """Extract information about embedding conversion candidates"""
        conversion_candidates_data = []
        for c in pipe.components:
            if ComponentUtils.component_has_embeddings_requirement(c) and not PipeUtils.is_trainable_pipe(pipe):
                storage_ref = StorageRefUtils.extract_storage_ref(c)
                conversion_applicable, conversion_data = PipelineQueryVerifier.check_if_storage_ref_is_satisfied_or_get_conversion_candidate(
                    c, pipe, storage_ref)
                if conversion_applicable: conversion_candidates_data.append(conversion_data)

        return conversion_candidates_data

    @staticmethod
    def get_missing_required_features(pipe):
        """For every component_to_resolve in the pipeline"""
        provided_features_no_ref = ComponentUtils.clean_irrelevant_features(
            PipelineQueryVerifier.extract_provided_features_refless_from_pipe(pipe))
        required_features_no_ref = ComponentUtils.clean_irrelevant_features(
            PipelineQueryVerifier.extract_required_features_refless_from_pipe(pipe))
        provided_features_ref = ComponentUtils.clean_irrelevant_features(
            PipelineQueryVerifier.extract_provided_features_ref_from_pipe(pipe))
        required_features_ref = ComponentUtils.clean_irrelevant_features(
            PipelineQueryVerifier.extract_required_features_ref_from_pipe(pipe))

        is_trainable = PipeUtils.is_trainable_pipe(pipe)
        conversion_candidates = PipelineQueryVerifier.extract_sentence_embedding_conversion_candidates(
            pipe)
        pipe.has_trainable_components = is_trainable

        required_features_ref, conversion_candidates = PipeUtils.remove_convertable_storage_refs(required_features_ref,
                                                                                                 conversion_candidates,
                                                                                                 provided_features_ref)
        provided_features_ref, required_features_ref = PipeUtils.update_converter_storage_refs_and_cols(pipe,
                                                                                                        provided_features_ref,
                                                                                                        required_features_ref)

        if is_trainable:

            trainable_index, embed_type = PipeUtils.find_trainable_embed_consumer(pipe)

            required_features_ref = []
            if embed_type is not None:
                # After resolve for a word embedding ,we must fix all NONES and set their storage refs !
                # embed consuming trainable annotators get their storage ref set here
                if len(provided_features_ref) == 0:
                    required_features_no_ref.append(embed_type)
                    if embed_type == NLP_FEATURES.CHUNK_EMBEDDINGS:
                        required_features_no_ref.append(NLP_FEATURES.WORD_EMBEDDINGS)
                if len(provided_features_ref) >= 1 and embed_type == NLP_FEATURES.CHUNK_EMBEDDINGS:
                    # This case is for when 1 Embed is preloaded and we still need to load the converter
                    if any(NLP_FEATURES.WORD_EMBEDDINGS in c for c in provided_features_ref):
                        required_features_no_ref.append(embed_type)

            if len(provided_features_ref) >= 1:
                # TODO Appraoches / Trainable models have no setStorageRef, we must set it after fitting
                pipe.components[trainable_index].storage_ref = provided_features_ref[0].split('@')[-1]

        missing_features_no_ref = set(required_features_no_ref) - set(
            provided_features_no_ref)  # - set(['text','label'])
        missing_features_ref = set(required_features_ref) - set(provided_features_ref)

        PipelineQueryVerifier.log_resolution_status(provided_features_no_ref, required_features_no_ref,
                                                    provided_features_ref, required_features_ref, is_trainable,
                                                    conversion_candidates, missing_features_no_ref,
                                                    missing_features_ref, )
        return missing_features_no_ref, missing_features_ref, conversion_candidates

    @staticmethod
    def add_sentence_embedding_converter(resolution_data: StorageRefConversionResolutionData) -> NluComponent:
        """ Return a Word to Sentence Embedding converter for a given Component. The input cols with match the Sentence Embedder ones
            The converter is a NLU Component Embelishement of the Spark NLP Sentence Embeddings Annotator
        """
        logger.info(f'Adding Sentence embedding conversion for Embedding Provider={resolution_data}')
        word_embedding_provider = resolution_data.component_candidate
        c = jsl_id_to_empty_component(NLP_NODE_IDS.SENTENCE_EMBEDDINGS_CONVERTER)
        storage_ref = StorageRefUtils.extract_storage_ref(word_embedding_provider)
        c.set_metadata(c.get_default_model(), 'sentence_embedding_converter',
                       NLP_NODE_IDS.SENTENCE_EMBEDDINGS_CONVERTER, 'xx', False, Licenses.open_source, storage_ref)
        c.model.setStorageRef(storage_ref)
        # set output cols
        embed_AT_out = NLP_FEATURES.SENTENCE_EMBEDDINGS + '@' + storage_ref
        c.model.setOutputCol(embed_AT_out)
        c.spark_output_column_names = [embed_AT_out]
        c.spark_input_column_names = [NLP_FEATURES.DOCUMENT, NLP_FEATURES.WORD_EMBEDDINGS + '@' + storage_ref]
        c.model.setInputCols(c.spark_input_column_names)
        return c

    @staticmethod
    def add_chunk_embedding_converter(
            resolution_data: StorageRefConversionResolutionData) -> NluComponent:
        """ Return a Word to CHUNK Embedding converter for a given Component. The input cols with match the Sentence Embedder ones
            The converter is a NLU Component Embelishement of the Spark NLP Sentence Embeddings Annotator
            The CHUNK embedder requires entities and also embeddings to generate data from. Since there could be multiple entities generators, we neeed to pass the correct one
        """
        # TODO REFACTOR
        logger.info(f'Adding Chunk embedding conversion  Provider={resolution_data} and NER Converter provider = ')
        word_embedding_provider = resolution_data.component_candidate
        entities_col = 'entities'
        embed_provider_col = word_embedding_provider.info.spark_output_column_names[0]

        c = jsl_id_to_empty_component(NLP_NODE_IDS.CHUNK_EMBEDDINGS_CONVERTER)
        c.set_metadata(c.get_default_model(),
                       NLP_NODE_IDS.CHUNK_EMBEDDINGS_CONVERTER, NLP_NODE_IDS.CHUNK_EMBEDDINGS_CONVERTER,
                       'xx',
                       False, Licenses.open_source)

        # c = nlu.embeddings_chunker.EmbeddingsChunker(annotator_class='chunk_embedder')
        storage_ref = StorageRefUtils.extract_storage_ref(word_embedding_provider)
        c.model.setStorageRef(storage_ref)
        c.info.storage_ref = storage_ref

        c.model.setInputCols(entities_col, embed_provider_col)
        c.model.setOutputCol('chunk_embeddings@' + storage_ref)
        c.info.spark_input_column_names = [entities_col, embed_provider_col]
        c.info.input_column_names = [entities_col, embed_provider_col]

        c.info.spark_output_column_names = ['chunk_embeddings@' + storage_ref]
        c.info.output_column_names = ['chunk_embeddings@' + storage_ref]
        return c

    @staticmethod
    def check_if_all_conversions_satisfied(components_for_embedding_conversion):
        """Check if all dependencies are satisfied."""
        for conversion in components_for_embedding_conversion:
            if conversion.component_candidate is not None: return False
        return True

    @staticmethod
    def check_dependencies_satisfied(missing_components, missing_storage_refs,
                                     components_for_embedding_conversion):
        """Check if all dependencies are satisfied."""
        return len(missing_components) == 0 and len(
            missing_storage_refs) == 0 and PipelineQueryVerifier.check_if_all_conversions_satisfied(
            components_for_embedding_conversion)

    @staticmethod
    def has_licensed_components(pipe) -> bool:
        """Check if any licensed components in component_list"""
        for c in pipe.components:
            if c.license == Licenses.hc or c.license == Licenses.ocr:
                return True
        return False

    @staticmethod
    def check_same_as_last_iteration(last_missing_components, last_missing_storage_refs,
                                     last_components_for_embedding_conversion, missing_components, missing_storage_refs,
                                     components_for_embedding_conversion):
        return last_missing_components == missing_components and last_missing_storage_refs == missing_storage_refs and last_components_for_embedding_conversion == components_for_embedding_conversion
    @staticmethod
    def except_infinity_loop(reason):
        raise Exception(f"Sorry, nlu has problems building this spell, please report this issue. Problem={reason}")

    @staticmethod
    def satisfy_dependencies(pipe):
        """Feature Dependency Resolution Algorithm.
         For a given pipeline with N components, builds a DAG in reverse and satisfy each of their dependencies and child dependencies
         with a BFS approach and returns the resulting pipeline
         :param pipe: Nlu Pipe containing components for which dependencies should be satisfied
         :return: Nlu pipe with dependencies satisfied
         """
        all_features_provided = False
        is_licensed = PipelineQueryVerifier.has_licensed_components(pipe)
        pipe.has_licensed_components = is_licensed
        is_trainable = PipeUtils.is_trainable_pipe(pipe)

        loop_count = 0
        max_loop_count = 5

        while all_features_provided == False:
            # After new components have been added, check again for the new components if requriements are met
            components_to_add = []
            missing_components, missing_storage_refs, components_for_embedding_conversion = \
                PipelineQueryVerifier.get_missing_required_features(pipe)
            if PipelineQueryVerifier.check_dependencies_satisfied(missing_components, missing_storage_refs,
                                                                  components_for_embedding_conversion):
                # Now all features are provided
                break

            # Update last iteration variables
            last_missing_components, last_missing_storage_refs, last_components_for_embedding_conversion = missing_components, missing_storage_refs, components_for_embedding_conversion
            # Create missing base storage ref producers, i.e. embeddings
            for missing_component in missing_storage_refs:
                component = resolve_feature(missing_component, language=pipe.lang,
                                            is_licensed=is_licensed, is_trainable_pipe=is_trainable)
                if component is None:
                    continue
                if 'chunk_emb' in missing_component:
                    components_to_add.append(ComponentUtils.config_chunk_embed_converter(component))
                else:
                    components_to_add.append(component)

            # Create missing base components, storage refs are fetched in previous loop
            for missing_component in missing_components:
                components_to_add.append(
                    resolve_feature(missing_component, language=pipe.lang, is_licensed=is_licensed,
                                    is_trainable_pipe=is_trainable))

            # Create embedding converters
            for resolution_info in components_for_embedding_conversion:
                converter = None
                if 'word2chunk' == resolution_info.type:
                    converter = PipelineQueryVerifier.add_chunk_embedding_converter(resolution_info)
                elif 'word2sentence' == resolution_info.type:
                    converter = PipelineQueryVerifier.add_sentence_embedding_converter(resolution_info)
                if converter is not None:
                    components_to_add.append(converter)

            logger.info(f'Resolved for missing components the following NLU components : {components_to_add}')

            # Add missing components
            for new_component in components_to_add:
                if  new_component:
                    logger.info(f'adding {new_component.name}')
                    pipe.add(new_component)

            # For some models we update storage ref to the resovling models storageref.
            # We need to update them so dependencies can properly be deducted as satisfied
            pipe = PipeUtils.update_bad_storage_refs(pipe)

            # Check if we are in an infinity loop
            if PipelineQueryVerifier.check_same_as_last_iteration(last_missing_components, last_missing_storage_refs,
                                                                  last_components_for_embedding_conversion,
                                                                  missing_components, missing_storage_refs,
                                                                  components_for_embedding_conversion):
                loop_count += 1
            else:
                loop_count = 0
            if loop_count > max_loop_count:
                PipelineQueryVerifier.except_infinity_loop('Failure resolving feature dependencies')

        logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.info(f"ALL DEPENDENCIES SATISFIED")
        return pipe

    @staticmethod
    def check_and_fix_component_output_column_name_satisfaction(pipe):
        '''
        This function verifies that every input and output column name of a component_to_resolve is satisfied.
        If some output names are missing, it will be added by this method.
        Usually classifiers need to change their input column name, so that it matches one of the previous embeddings because they have dynamic output names
        This function performs the following steps :
        1. For each component_to_resolve we verify that all input column names are satisfied  by checking all other components output names
        2. When a input column is missing we do the following :
        2.1 Figure out the type of the missing input column. The name of the missing column should be equal to the type
        2.2 Check if there is already a component_to_resolve in the component_list, which provides this input (It should)
        2.3. When A providing component_to_resolve is found, check if storage ref matches up.
        2.4 If True for all, update provider component_to_resolve output name, or update the original component_to_resolve input name
        :return: NLU pipeline where the output and input column names of the models have been adjusted to each other
        '''
        logger.info("Fixing input and output column names")
        for component_to_check in pipe.components:
            if component_to_check.loaded_from_pretrained_pipe: continue
            input_columns = set(component_to_check.spark_input_column_names)
            # a component_to_resolve either has '' storage ref or at most 1
            logger.info(
                f'Checking for component_to_resolve {component_to_check.name} wether inputs {input_columns} is satisfied by another component_to_resolve in the component_list ', )
            for other_component in pipe.components:
                if component_to_check.name == other_component.name: continue
                output_columns = set(other_component.spark_output_column_names)
                input_columns -= output_columns  # remove provided columns

            input_columns = ComponentUtils.clean_irrelevant_features(input_columns)

            # Resolve basic mismatches, usually storage refs
            if len(input_columns) != 0 and not pipe.has_trainable_components or ComponentUtils.is_embedding_consumer(
                    component_to_check):  # fix missing column name
                # We must not only check if input satisfied, but if storage refs match! and Match Storage_refs accordingly
                logger.info(f"Fixing bad input col for C={component_to_check} untrainable component_list")
                resolved_storage_ref_cols = []
                for missing_column in input_columns:
                    for other_component in pipe.components:
                        if component_to_check.name == other_component.name: continue
                        if other_component.type == missing_column:
                            # We update the output name for the component_to_resolve which consumes our feature
                            if StorageRefUtils.has_storage_ref(
                                    other_component) and ComponentUtils.is_embedding_provider(component_to_check):
                                if ComponentUtils.are_producer_consumer_matches(component_to_check, other_component):
                                    resolved_storage_ref_cols.append(
                                        (other_component.spark_output_column_names[0], missing_column))

                            component_to_check.spark_output_column_names = [missing_column]
                            logger.info(
                                f'Resolved requirement for missing_column={missing_column} with inputs from provider={other_component.name} by col={missing_column} ')
                            other_component.model.setOutputCol(missing_column)

                for resolution, unsatisfied in resolved_storage_ref_cols:
                    component_to_check.spark_input_column_names.remove(unsatisfied)
                    component_to_check.spark_input_column_names.append(resolution)



            # Resolve training missmatches
            elif len(input_columns) != 0 and pipe.has_trainable_components:  # fix missing column name
                logger.info(f"Fixing bad input col for C={component_to_check} trainable component_list")
                # for trainable components, we change their input columns and leave other components outputs unchanged
                for missing_column in input_columns:
                    for other_component in pipe.components:
                        if component_to_check.name == other_component.name: continue
                        if other_component.type == missing_column:
                            # We update the input col name for the componenet that has missing cols
                            component_to_check.spark_input_column_names.remove(missing_column)
                            component_to_check.spark_input_column_names.append(
                                other_component.spark_output_column_names[0])
                            component_to_check.model.setInputCols(
                                component_to_check.spark_input_column_names)

                            logger.info(
                                f'Setting input col columns for component_to_resolve {component_to_check.name} to {other_component.spark_output_column_names[0]} ')

        return pipe

    @staticmethod
    def check_and_fix_nlu_pipeline(pipe):
        """Check if the NLU pipeline is ready to transform data and return it.
        If all dependencies not satisfied, returns a new NLU pipeline where dependencies and sub-dependencies are satisfied.
        Checks and resolves in the following order :
        1. Get a reference list of input features missing for the current component_list
        2. Resolve the list of missing features by adding new  Annotators to component_list
        3. Add NER Converter if required (When there is a NER model_anno_obj)
        4. Fix order and output column names
        5.

        :param pipe:
        :return:
        """
        # main entry point for Model stacking withouth pretrained pipelines
        # requirements and provided features will be lists of lists

        # 0. Clean old @AT storage ref from all columns
        # logger.info('Cleaning old AT refs')
        # pipe = PipeUtils.clean_AT_storage_refs(pipe)

        # 1. Resolve dependencies, builds a DAG in reverse and satisfies dependencies with a Breadth-First-Search approach
        # 0. Write additional metadata to the pipe pre pipe construction
        pipe = PipeUtils.add_metadata_to_pipe(pipe)

        logger.info('Satisfying dependencies')
        pipe = PipelineQueryVerifier.satisfy_dependencies(pipe)

        # 2. Enforce naming schema <col_name>@<storage_ref> for storage_ref consumers and producers and <entity@nlu_ref> and <ner@nlu_ref> for NER and NER-Converters
        # and add NER-IOB to NER-Pretty converters for every NER model_anno_obj that is not already feeding a NER converter
        pipe = PipeUtils.enforce_AT_schema_on_pipeline_and_add_NER_converter(pipe)

        # 2.1 If Sentence Resolvers are in pipeline, all Sentence-Embeddings must feed from Chunk2Doc which stems from the entities column to resolve
        pipe = PipelineQueryVerifier.enforce_chunk2doc_on_sentence_embeddings(pipe)

        # 3. Validate naming of output columns is correct and no error will be thrown in spark
        logger.info('Fixing column names')
        pipe = PipelineQueryVerifier.check_and_fix_component_output_column_name_satisfaction(pipe)

        # 4. Set on every NLP Annotator the output columns
        pipe = PipeUtils.enforce_NLU_columns_to_NLP_columns(pipe)

        # 5. fix order
        logger.info('Optimizing component_list component_to_resolve order')
        pipe = PipelineQueryVerifier.check_and_fix_component_order(pipe)

        # 6. Rename overlapping/duplicate leaf columns in the DAG
        logger.info('Renaming duplicates cols')
        pipe = PipeUtils.rename_duplicate_cols(pipe)

        # 7. enfore again because trainable pipes might mutate component_list cols
        pipe = PipeUtils.enforce_NLU_columns_to_NLP_columns(pipe)

        # 8. Write additional metadata to the pipe post pipe construction
        pipe = PipeUtils.add_metadata_to_pipe(pipe)

        logger.info('Done with component_list optimizing')

        return pipe

    @staticmethod
    def check_and_fix_component_order(pipe):
        '''
        This method takes care that the order of components is the correct in such a way,that the pipeline can be iteratively processed by spark NLP.
        Column Names will not be touched. DAG Task Sort basically.
        '''
        logger.info("Starting to optimize component_to_resolve order ")

        correct_order_component_pipeline = []
        provided_features = []
        all_components_ordered = False
        unsorted_components = pipe.components
        update_last_type = False
        last_type_sorted = None
        trainable_updated = False
        pipe.components = sorted(pipe.components, key=lambda x: x.type)
        if not pipe.contains_ocr_components:
            # if OCR we must take text sorting into account. Non-OCR pipes get text provided externalyl
            provided_features.append('text')

        loop_count = 0
        max_loop_count = 10*len(pipe.components)
        last_correct_order_component_pipeline = []
        last_provided_features = []

        while not all_components_ordered:
            if update_last_type:
                last_type_sorted = None
            else:
                update_last_type = True
            for component in unsorted_components:
                logger.info(f"Optimizing order for component_to_resolve {component.name}")
                input_columns = ComponentUtils.remove_storage_ref_from_features(
                    ComponentUtils.clean_irrelevant_features(component.spark_input_column_names.copy(), False, False))
                if last_type_sorted is None or component.type == last_type_sorted:
                    if set(input_columns).issubset(provided_features):
                        correct_order_component_pipeline.append(component)

                        # Leave pretrained component_list components untouched
                        if component.loaded_from_pretrained_pipe:
                            unsorted_components.remove(component)
                        if component in unsorted_components:
                            unsorted_components.remove(component)

                        # TODO remove storage ref from provided features ?
                        provided_features += ComponentUtils.remove_storage_ref_from_features(
                            ComponentUtils.clean_irrelevant_features(component.spark_output_column_names.copy(), False,
                                                                     False))

                        last_type_sorted = component.type
                        update_last_type = False

                        break

            if len(unsorted_components) == 0:
                all_components_ordered = True

            if not all_components_ordered and len(
                    unsorted_components) <= 2 and pipe.has_trainable_components and not trainable_updated and \
                    unsorted_components[0].trainable and 'sentence_embeddings@' in unsorted_components[
                0].spark_input_column_names:
                # special case, if trainable then we feed embed consumers on the first sentence embed provider
                # 1. Find first sent embed provider
                # 2. substitute any 'sent_embed@' consumer inputs for the provider col
                for f in provided_features:
                    if 'sentence_embeddings' in f and not trainable_updated:
                        unsorted_components[0].spark_input_column_names.remove('sentence_embeddings@')
                        if 'sentence_embeddings@' in unsorted_components[0].spark_input_column_names:
                            unsorted_components[0].spark_input_column_names.remove('sentence_embeddings@')
                        unsorted_components[0].spark_input_column_names.append(f)
                        if f not in unsorted_components[0].spark_input_column_names:  unsorted_components[
                            0].spark_input_column_names.append(f)
                        trainable_updated = True

            if not all_components_ordered and len(
                    unsorted_components) <= 2 and pipe.has_trainable_components and not trainable_updated and \
                    unsorted_components[0].trainable and 'word_embeddings@' in unsorted_components[
                0].spark_input_column_names:
                # special case, if trainable then we feed embed consumers on the first sentence embed provider
                # 1. Find first sent embed provider
                # 2. substitute any 'sent_embed@' consumer inputs for the provider col
                for f in provided_features:
                    if 'word_embeddings' in f and not trainable_updated:
                        unsorted_components[0].spark_input_column_names.remove('word_embeddings@')
                        if 'word_embeddings@' in unsorted_components[0].spark_input_column_names:  unsorted_components[
                            0].spark_input_column_names.remove(
                            'word_embeddings@')
                        unsorted_components[0].spark_input_column_names.append(f)
                        if f not in unsorted_components[0].spark_input_column_names:  unsorted_components[
                            0].spark_input_column_names.append(f)
                        trainable_updated = True

            # detect endless loop
            if last_correct_order_component_pipeline == correct_order_component_pipeline and last_provided_features == provided_features :
                loop_count +=1
            else:
                loop_count = 0
            if loop_count > max_loop_count:
                PipelineQueryVerifier.except_infinity_loop('Failure sorting dependencies')
            last_provided_features = provided_features.copy()
            # correct_order_component_pipeline = last_correct_order_component_pipeline.copy()
            last_correct_order_component_pipeline = correct_order_component_pipeline.copy()

        pipe.components = correct_order_component_pipeline


        return pipe

    @staticmethod
    def is_storage_ref_match(embedding_consumer, embedding_provider, pipe):
        """Check for 2 components, if one provides the embeddings for the other. Makes sure that pipe_prediction_output_level matches up (chunk/sent/tok/embeds)"""
        consumer_AT_ref = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(embedding_consumer, 'input')
        provider_AT_rev = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(embedding_provider, 'output')
        consum_level = ComponentUtils.extract_embed_level_identity(embedding_consumer, 'input')
        provide_level = ComponentUtils.extract_embed_level_identity(embedding_provider, 'output')

        consumer_ref = StorageRefUtils.extract_storage_ref(embedding_consumer)
        provider_rev = StorageRefUtils.extract_storage_ref(embedding_provider)

        # input/output levels must match
        if consum_level != provide_level: return False

        # If storage ref dont match up, we must consult the storage_ref_2_embed mapping if it still maybe is a match, otherwise it is not.
        if consumer_ref == provider_rev: return True

        # Embed Components have have been resolved via@ have a  nlu_resolution_ref_source will match up with the consumer ref if correct embedding.
        if hasattr(embedding_provider.info, 'nlu_ref'):
            if consumer_ref == StorageRefUtils.extract_storage_ref(embedding_provider.info.nlu_ref): return True

        # If it is either  sentence_embedding_converter or chunk_embedding_converter then we gotta check what the storage ref of the inpot of those is.
        # If storage ref matches up, the providers output will match the consumer
        # if embedding_provider
        if embedding_provider.info.name in ["chunk_embedding_converter",
                                            'sentence_embedding_converter']:  # TODO FOR RESOLUTION
            nlu_ref, conv_prov_storage_ref = PipelineQueryVerifier.get_converters_provider_info(embedding_provider,
                                                                                                pipe)

        return False

    @staticmethod
    def is_matching_level(embedding_consumer, embedding_provider):
        """Check for embedding consumer if input level matches up outputlevel of consumer
        """

    @staticmethod
    def get_converters_provider_info(embedding_provider, pipe):
        """For a component_to_resolve and a component_list, find storage_ref and """

    @staticmethod
    def enforce_chunk2doc_on_sentence_embeddings(pipe):
        """
        #If Sentence Resolvers are in pipeline, all Sentence-Embeddings must feed from Chunk2Doc which stems from
        the entities column to resolve We need to update input/output types of sentence Resolver, to the component_to_resolve
        so sorting does not get confused
        """
        if not pipe.has_licensed_components:
            return pipe
        resolvers = []
        ner_converters = []
        sentence_embeddings = []
        # Find Resolver
        for i, c in enumerate(pipe.components):
            if c.loaded_from_pretrained_pipe: continue
            # if isinstance(c.model_anno_obj, SentenceEntityResolverModel): resolvers.append(c)
            # if isinstance(c.model_anno_obj, (NerConverter, NerConverterInternal)): ner_converters.append(c)
            # if 'sentence_embeddings' == c.info.type: sentence_embeddings.append(c)
            if c.name == NLP_HC_NODE_IDS.SENTENCE_ENTITY_RESOLVER:
                resolvers.append(c)
            if c.name in [NLP_NODE_IDS.NER_CONVERTER, NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL]:
                ner_converters.append(c)
            if c.type == AnnoTypes.DOCUMENT_EMBEDDING or c.type == AnnoTypes.SENTENCE_EMBEDDING:
                sentence_embeddings.append(c)

        # No resolvers, nothing to update
        if len(resolvers) == 0:
            return pipe

        # Update Resolver
        # TODO this does not work in multi resolver scenarios reliably
        if NLP_FEATURES.DOCUMENT in sentence_embeddings[0].in_types:
            sentence_embeddings[0].in_types.remove(NLP_FEATURES.DOCUMENT)
        if NLP_FEATURES.SENTENCE in sentence_embeddings[0].in_types:
            sentence_embeddings[0].in_types.remove(NLP_FEATURES.SENTENCE)
        if NLP_FEATURES.DOCUMENT in sentence_embeddings[0].spark_input_column_names:
            sentence_embeddings[0].spark_input_column_names.remove(NLP_FEATURES.DOCUMENT)
        if NLP_FEATURES.SENTENCE in sentence_embeddings[0].spark_input_column_names:
            sentence_embeddings[0].spark_input_column_names.remove(NLP_FEATURES.SENTENCE)
        sentence_embeddings[0].in_types.append(NLP_FEATURES.DOCUMENT_FROM_CHUNK)
        sentence_embeddings[0].spark_input_column_names.append(NLP_FEATURES.DOCUMENT_FROM_CHUNK)

        # sentence_embeddings[0].info.inputs = ['chunk2doc']
        # sentence_embeddings[0].info.spark_input_column_names = ['chunk2doc']
        # sentence_embeddings[0].model_anno_obj.setInputCols('chunk2doc') # shouldb e handled by enforcing
        # chunk2doc.model_anno_obj.setOutputCol("chunk2doc")
        # chunk2doc.info.inputs = ner_converters[0].spark_output_column_names

        # TODO this will not be resolved by the resolution Algo!!
        chunk2doc = resolve_feature(NLP_FEATURES.DOCUMENT_FROM_CHUNK, 'xx')
        chunk2doc.model.setInputCols(ner_converters[0].spark_output_column_names)
        chunk2doc.spark_input_column_names = ner_converters[0].spark_output_column_names
        pipe.components.append(chunk2doc)
        # this will add a entity converter and a NER model_anno_obj if none provided
        pipe = PipelineQueryVerifier.satisfy_dependencies(pipe)

        return pipe

    @staticmethod
    def log_resolution_status(provided_features_no_ref, required_features_no_ref, provided_features_ref,
                              required_features_ref, is_trainable, conversion_candidates, missing_features_no_ref,
                              missing_features_ref, ):
        logger.info(f"========================================================================")
        logger.info(f"Resolution Status provided_features_no_ref = {set(provided_features_no_ref)}")
        logger.info(f"Resolution Status required_features_no_ref = {set(required_features_no_ref)}")
        logger.info(f"Resolution Status provided_features_ref    = {set(provided_features_ref)}")
        logger.info(f"Resolution Status required_features_ref    = {set(required_features_ref)}")
        logger.info(f"Resolution Status is_trainable             = {is_trainable}")
        logger.info(f"Resolution Status conversion_candidates    = {conversion_candidates}")
        logger.info(f"Resolution Status missing_features_no_ref  = {set(missing_features_no_ref)}")
        logger.info(f"Resolution Status conversion_candidates    = {set(missing_features_ref)}")
        logger.info(f"========================================================================")
