import nlu
import logging
logger = logging.getLogger('nlu')
from sparknlp.annotator import *
from nlu.pipe.pipe_components import SparkNLUComponent
from nlu.pipe.utils.pipe_utils import PipeUtils
from nlu.pipe.utils.component_utils import ComponentUtils
from nlu.pipe.utils.storage_ref_utils import StorageRefUtils

from dataclasses import dataclass
from nlu.pipe.component_resolution import get_default_component_of_type

@dataclass
class StorageRefConversionResolutionData:
    """Hold information that can be used to resolve to a NLU component, which satisfies the storage ref demands."""
    storage_ref: str # storage ref a resolver component should have
    component_candidate: SparkNLUComponent # from which NLU component should the converter feed
    type: str # what kind of conversion, either word2chunk or word2sentence

class PipelineQueryVerifier():
    '''
        Pass a list of NLU components to the pipeline (or a NLU pipeline)
        For every component, it checks if all requirements are met.
        It checks and fixes the following issues  for a list of components:
        1. Missing Features / component requirements
        2. Bad order of components (which will cause missing features.
        3. Check Feature naems in the output
        4. Check wether pipeline needs to be fitted
    '''
    @staticmethod
    def check_if_storage_ref_is_satisfied_or_get_conversion_candidate(component_to_check:SparkNLUComponent, pipe, storage_ref_to_find:str):
        """Check if any other component in the pipeline has same storage ref as the input component..
        Returns 1. If
        If there is a candiate but it has different level, it will be returned as candidate

        If first condition is not satified, consults the namespace.storage_ref_2_nlp_ref

        """
        # If there is just 1 component, there is nothing to check
        if len(pipe.components) == 1: return False, None
        conversion_candidate = None
        conversion_type = "no_conversion"
        logger.info(f'checking for storage={storage_ref_to_find} is avaiable in pipe..')
        for c in pipe.components:
            if component_to_check.info.name != c.info.name:
                if StorageRefUtils.has_storage_ref(c):
                    if StorageRefUtils.extract_storage_ref(c) == storage_ref_to_find:
                        # Both components have Different Names AND their Storage Ref Matches up AND they both take in tokens -> Match
                        if 'token' in component_to_check.info.inputs and c.info.type == 'word_embeddings':
                            logger.info(f'Word Embedding Match found = {c.info.name}')
                            return False, None

                        # Since document and be substituted for sentence and vice versa if either of them matches up we have a match
                        if 'sentence_embeddings' in component_to_check.info.inputs and c.info.type == 'sentence_embeddings':
                            logger.info(f'Sentence Emebdding Match found = {c.info.name}')
                            return False, None

                        # component_to_check requires Sentence_embedding but the Matching Storage_ref component takes in Token
                        #   -> Convert the Output of the Match to SentenceLevel and feed the component_to_check to the new component
                        if 'sentence_embeddings' in component_to_check.info.inputs and c.info.type == 'word_embeddings':
                            logger.info(f'Sentence Embedding Conversion Candidate found={c.info.name}')
                            conversion_type      = 'word2sentence'
                            conversion_candidate = c


                        #analogus case as above for chunk
                        if 'chunk_embeddings' in component_to_check.info.inputs and c.info.type == 'word_embeddings':
                            logger.info(f'Sentence Embedding Conversion Candidate found={c.info.name}')
                            conversion_type      = 'word2chunk'
                            conversion_candidate = c

        logger.info(f'No matching storage ref found')
        return True, StorageRefConversionResolutionData(storage_ref_to_find, conversion_candidate, conversion_type)
    @staticmethod
    def extract_required_features_refless_from_pipe(pipe):
        """Extract provided features from pipe, which have no storage ref"""
        provided_features_no_ref = []
        for c in pipe.components:
            for feat in c.info.inputs:
                if 'embed' not in feat : provided_features_no_ref.append(feat)
        return ComponentUtils.clean_irrelevant_features(provided_features_no_ref)
    @staticmethod
    def extract_provided_features_refless_from_pipe(pipe):
        """Extract provided features from pipe, which have no storage ref"""
        provided_features_no_ref = []
        for c in pipe.components:
            for feat in c.info.outputs:
                if 'embed' not in feat : provided_features_no_ref.append(feat)
        return  ComponentUtils.clean_irrelevant_features(provided_features_no_ref)
    @staticmethod
    def extract_provided_features_ref_from_pipe(pipe):
        """Extract provided features from pipe, which have  storage ref"""
        provided_features_ref = []
        for c in pipe.components:
            for feat in c.info.outputs:
                if 'embed' in feat :
                    if '@' not in feat  : provided_features_ref.append(feat +"@"+ StorageRefUtils.extract_storage_ref(c))
                    else  : provided_features_ref.append(feat)
        return ComponentUtils.clean_irrelevant_features(provided_features_ref)
    @staticmethod
    def extract_required_features_ref_from_pipe(pipe):
        """Extract provided features from pipe, which have  storage ref"""
        provided_features_ref = []
        for c in pipe.components:
            for feat in c.info.inputs:
                if 'embed' in feat :
                    # if StorageRefUtils.extract_storage_ref(c) !='':  # special edge case, some components might not have a storage ref set
                    if '@' not in feat : provided_features_ref.append(feat +"@"+ StorageRefUtils.extract_storage_ref(c))
                    else  : provided_features_ref.append(feat)

        return ComponentUtils.clean_irrelevant_features(provided_features_ref)
    @staticmethod
    def extract_sentence_embedding_conversion_candidates(pipe):
        """Extract information about embedding conversion candidates"""
        conversion_candidates_data = []
        for c in pipe.components:
            if ComponentUtils.component_has_embeddings_requirement(c) and not PipeUtils.is_trainable_pipe(pipe):
                storage_ref = StorageRefUtils.extract_storage_ref(c)
                conversion_applicable, conversion_data = PipelineQueryVerifier.check_if_storage_ref_is_satisfied_or_get_conversion_candidate(c, pipe, storage_ref)
                if conversion_applicable: conversion_candidates_data.append(conversion_data)

        return conversion_candidates_data
    @staticmethod
    def get_missing_required_features(pipe):
        """For every component in the pipeline"""
        provided_features_no_ref                = ComponentUtils.clean_irrelevant_features(PipelineQueryVerifier.extract_provided_features_refless_from_pipe(pipe))
        required_features_no_ref                = ComponentUtils.clean_irrelevant_features(PipelineQueryVerifier.extract_required_features_refless_from_pipe(pipe))
        provided_features_ref                   = ComponentUtils.clean_irrelevant_features(PipelineQueryVerifier.extract_provided_features_ref_from_pipe(pipe))
        required_features_ref                   = ComponentUtils.clean_irrelevant_features(PipelineQueryVerifier.extract_required_features_ref_from_pipe(pipe))
        is_trainable                            = PipeUtils.is_trainable_pipe(pipe)
        conversion_candidates                   = PipelineQueryVerifier.extract_sentence_embedding_conversion_candidates(pipe)
        pipe.has_trainable_components           = is_trainable
        if is_trainable:
            trainable_index, embed_type = PipeUtils.find_trainable_embed_consumer(pipe)

            required_features_ref = []
            if embed_type is not None :
                #TODO storage ref of chunk embeddigns and chunk embed consumers !?!
                # TODO if we have a TRAINABLE CHUNK-CONSUMEr, we first will have chunk_embed@NONE, word_embed@NONE, word_embed@SOME,
                # After resolve for a word embedding ,we must fix all NONES and set their storage refs !
                # embed consuming trainablea nnotators get their storage ref set here
                if len(provided_features_ref) == 0 :
                    required_features_no_ref.append(embed_type)
                    if embed_type=='chunk_embeddings': required_features_no_ref.append('word_embeddings')
                if len(provided_features_ref) == 1  and embed_type=='chunk_embeddings' :
                    # This case is for when 1 Embed is preloaded and we still need to load the converter
                    if any('word_embedding' in c for  c in provided_features_ref) :required_features_no_ref.append(embed_type)


                else :
                    #set storage ref
                    if embed_type=='chunk_embeddings' and len(provided_features_ref) >  1 :
                        # TODO UPDATE HERE ALL REFS TO THE ONE THATS NOT NONE !!!
                        training_storage_ref = ''
                        for c in pipe.components:
                            if c.info.type =='word_embeddings': training_storage_ref = StorageRefUtils.extract_storage_ref(c)
                        for c in pipe.components:
                            if c.info.type =='chunk_embeddings' :
                                c.info.storage_ref               = training_storage_ref
                                c.info.inputs                    = ['entities',f'word_embeddings@{training_storage_ref}']
                                c.info.spark_input_column_names  = ['entities',f'word_embeddings@{training_storage_ref}']
                                c.info.outputs                   = [f'chunk_embeddings@{training_storage_ref}']
                                c.info.spark_output_column_names = [f'chunk_embeddings@{training_storage_ref}']

                            if c.info.name =='chunk_resolver' :
                                c.info.storage_ref = training_storage_ref
                                c.info.inputs                    = ['token',f'chunk_embeddings@{training_storage_ref}']
                                c.info.spark_input_column_names  = ['token',f'chunk_embeddings@{training_storage_ref}']


                    elif len(provided_features_ref) >=1 :
                        pipe.components[trainable_index].info.storage_ref = provided_features_ref[0].split('@')[-1]

        components_for_ner_conversion = [] #

        missing_features_no_ref                 = set(required_features_no_ref) - set(provided_features_no_ref)# - set(['text','label'])
        missing_features_ref                    = set(required_features_ref)    - set(provided_features_ref)

        PipelineQueryVerifier.log_resolution_status(provided_features_no_ref,required_features_no_ref,provided_features_ref,required_features_ref,is_trainable,conversion_candidates,missing_features_no_ref,missing_features_ref,)
        return missing_features_no_ref,missing_features_ref, conversion_candidates
        # # default requirements so we can support all output levels. minimal extra comoputation effort. If we add CHUNK here, we will aalwayshave POS default
        # if not PipeUtils.check_if_component_is_in_pipe(pipe,'sentence',False ) : pipe_requirements.append(['sentence'])
        # if not PipeUtils.check_if_component_is_in_pipe(pipe,'token',False )    : pipe_requirements.append(['token'])



    @staticmethod
    def with_missing_ner_converters(pipe) :
        '''

        :param pipe: The pipeline wea dd NER converters to
        :return: new pipeline with NER converters added
        '''
        #
        # for component in pipe.components:
        #     if 'ner' in component.info.outputs:
        #         has_feeding_ner_converter = False
        #         for other_component in pipe.components:
        #             if  other_component.info.name == 'NerToChunkConverter' :
        #
        #
        #     pipe.add(get_default_component_of_type(('ner_converter')))
        #
        #

        return pipe

    @staticmethod
    def add_sentence_embedding_converter(resolution_data:StorageRefConversionResolutionData) -> SparkNLUComponent:
        """ Return a Word to Sentence Embedding converter for a given Component. The input cols with match the Sentence Embedder ones
            The converter is a NLU Component Embelishement of the Spark NLP Sentence Embeddings Annotator
        """
        logger.info(f'Adding Sentence embedding conversion for Embedding Provider={ resolution_data}')
        word_embedding_provider = resolution_data.component_candidate
        c = nlu.Util(annotator_class='sentence_embeddings')
        storage_ref = StorageRefUtils.extract_storage_ref(word_embedding_provider)
        # set storage rage
        c.model.setStorageRef(storage_ref)
        c.info.storage_ref = storage_ref

        #set output cols
        embed_AT_out = 'sentence_embeddings@' + storage_ref
        c.model.setOutputCol(embed_AT_out)
        c.info.spark_output_column_names = [embed_AT_out]
        c.info.outputs = [embed_AT_out]

        #set input cls
        c.model.setInputCols('document', )
        embed_provider_col = word_embedding_provider.info.spark_output_column_names[0]

        c.info.inputs = ['document', 'word_embeddings@'+storage_ref]
        c.info.spark_input_column_names = c.info.inputs
        c.model.setInputCols(c.info.inputs)

        word_embedding_provider.info.storage_ref = storage_ref
        return c

    @staticmethod
    def add_chunk_embedding_converter(resolution_data:StorageRefConversionResolutionData) -> SparkNLUComponent : # ner_converter_provider:SparkNLUComponent,
        """ Return a Word to CHUNK Embedding converter for a given Component. The input cols with match the Sentence Embedder ones
            The converter is a NLU Component Embelishement of the Spark NLP Sentence Embeddings Annotator
            The CHUNK embedder requires entities and also embeddings to generate data from. Since there could be multiple entities generators, we neeed to pass the correct one
        """
        logger.info(f'Adding Chunk embedding conversion for Embedding Provider={ resolution_data} and NER Converter provider = ')
        word_embedding_provider = resolution_data.component_candidate

        entities_col = 'entities' # ner_converter_provider.info.spark_output_column_names[0]
        embed_provider_col = word_embedding_provider.info.spark_output_column_names[0]

        c = nlu.embeddings_chunker.EmbeddingsChunker(annotator_class='chunk_embedder')
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
            if conversion.component_candidate is not None : return False
        return True

    @staticmethod
    def check_if_all_dependencies_satisfied(missing_components, missing_storage_refs, components_for_embedding_conversion):
        """Check if all dependencies are satisfied."""
        return len(missing_components) ==0 and len (missing_storage_refs) == 0 and PipelineQueryVerifier.check_if_all_conversions_satisfied(components_for_embedding_conversion)


    @staticmethod
    def has_licensed_components(pipe) -> bool:
        """Check if any licensed components in pipe"""
        for c in pipe.components :
            if c.info.license =='healthcare' : return True
        return False
    @staticmethod
    def satisfy_dependencies(pipe):
        """Dependency Resolution Algorithm.
        For a given pipeline with N components, builds a DAG in reverse and satisfiy each of their dependencies and child dependencies
         with a BFS approach and returns the resulting pipeline"""
        all_features_provided = False
        is_licensed = PipelineQueryVerifier.has_licensed_components(pipe)
        pipe.has_licensed_components=is_licensed
        is_trainable = PipeUtils.is_trainable_pipe(pipe)
        while all_features_provided == False:
            # After new components have been added, we must loop again and check for the new components if requriements are met
            components_to_add = []
            missing_components, missing_storage_refs, components_for_embedding_conversion = PipelineQueryVerifier.get_missing_required_features(pipe)
            logger.info(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            logger.info(f"Trying to resolve missing features for \n missing_components={missing_components} \n missing storage_refs={missing_storage_refs}\n conversion_candidates={components_for_embedding_conversion}")
            if PipelineQueryVerifier.check_if_all_dependencies_satisfied(missing_components, missing_storage_refs, components_for_embedding_conversion): break  # Now all features are provided


            # Create missing base storage ref producers, i.e embeddings
            for missing_component in missing_storage_refs:
                component = get_default_component_of_type(missing_component, language=pipe.lang, is_licensed=is_licensed, is_trainable_pipe=is_trainable)
                if component is None : continue
                if 'chunk_emb' in missing_component:
                    components_to_add.append(ComponentUtils.config_chunk_embed_converter(component))
                else :components_to_add.append(component)


            # Create missing base components, storage refs are fetched in rpevious loop
            for missing_component in missing_components:
                components_to_add.append(get_default_component_of_type(missing_component, language=pipe.lang,is_licensed=is_licensed, is_trainable_pipe=is_trainable))

            # Create embedding converters
            for resolution_info in components_for_embedding_conversion:
                converter=None
                if   'word2chunk' ==  resolution_info.type : converter =  PipelineQueryVerifier.add_chunk_embedding_converter(resolution_info)
                elif 'word2sentence' ==  resolution_info.type : converter = PipelineQueryVerifier.add_sentence_embedding_converter(resolution_info)
                if converter is not None: components_to_add.append(converter)



            logger.info(f'Resolved for missing components the following NLU components : {components_to_add}')

            # Add missing components
            for new_component in components_to_add:
                logger.info(f'adding {new_component.info.name}')
                pipe.add(new_component)



        logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.info(f"ALLL DEPENDENCIES SATISFIED")
        return pipe

    @staticmethod
    def check_and_fix_component_output_column_name_satisfaction(pipe):
        '''
        This function verifies that every input and output column name of a component is satisfied.
        If some output names are missing, it will be added by this methods.
        Usually classifiers need to change their input column name, so that it matches one of the previous embeddings because they have dynamic output names
        This function peforms the following steps :
        1. For each component we veryify that all input column names are satisfied  by checking all other components output names
        2. When a input column is missing we do the following :
        2.1 Figure out the type of the missing input column. The name of the missing column should be equal to the type
        2.2 Check if there is already a component in the pipe, which provides this input (It should)
        2.3. When A providing component is found, check if storage ref matches up.
        2.4 If True for all, update provider component output name, or update the original coponents input name
        :return: NLU pipeline where the output and input column names of the models have been adjusted to each other
        '''
        logger.info("Fixing input and output column names")
        # pipe = PipeUtils.enforce_AT_schema_on_pipeline(pipe)

        for component_to_check in pipe.components:
            input_columns = set(component_to_check.info.spark_input_column_names)
            # a component either has '' storage ref or at most 1
            logger.info(f'Checking for component {component_to_check.info.name} wether inputs {input_columns} is satisfied by another component in the pipe ', )
            for other_component in pipe.components:
                if component_to_check.info.name == other_component.info.name: continue
                output_columns = set(other_component.info.spark_output_column_names)
                input_columns -= output_columns # we substract alrfready provided columns

            input_columns = ComponentUtils.clean_irrelevant_features(input_columns)

            # Resolve basic mismatches, usually storage refs
            if len(input_columns) != 0 and not pipe.has_trainable_components or ComponentUtils.is_embedding_consumer(component_to_check):  # fix missing column name
                # We must not only check if input satisfied, but if storage refs match! and Match Storage_refs accordingly
                logger.info(f"Fixing bad input col for C={component_to_check} untrainable pipe")
                resolved_storage_ref_cols = []
                for missing_column in input_columns:
                    for other_component in pipe.components:
                        if component_to_check.info.name == other_component.info.name: continue
                        if other_component.info.type == missing_column:
                            # We update the output name for the component which consumes our feature

                            if StorageRefUtils.has_storage_ref(other_component) and ComponentUtils.is_embedding_provider(component_to_check):
                                if ComponentUtils.are_producer_consumer_matches(component_to_check,other_component):
                                    resolved_storage_ref_cols.append((other_component.info.spark_output_column_names[0],missing_column))

                            component_to_check.info.spark_output_column_names = [missing_column]
                            component_to_check.info.outputs = [missing_column]
                            logger.info(f'Resolved requirement for missing_column={missing_column} with inputs from provider={other_component.info.name} by col={missing_column} ')
                            other_component.model.setOutputCol(missing_column)


                for resolution, unsatisfied in resolved_storage_ref_cols:
                    component_to_check.info.spark_input_column_names.remove(unsatisfied)
                    component_to_check.info.spark_input_column_names.append(resolution)
                component_to_check.info.inputs = component_to_check.info.spark_input_column_names



            # TODO USE is_storage_ref_match ?
            # Resolve training missatches
            elif len(input_columns) != 0 and pipe.has_trainable_components:  # fix missing column name
                logger.info(f"Fixing bad input col for C={component_to_check} trainable pipe")

                # for trainable components, we change their input columns and leave other components outputs unchanged
                for missing_column in input_columns:
                    for other_component in pipe.components:
                        if component_to_check.info.name == other_component.info.name: continue
                        if other_component.info.type == missing_column:
                            # We update the input col name for the componenet that has missing cols
                            component_to_check.info.spark_input_column_names.remove(missing_column)
                            # component_to_check.component_info.inputs.remove(missing_column)
                            # component_to_check.component_info.inputs.remove(missing_column)
                            # component_to_check.component_info.inputs.append(other_component.component_info.spark_output_column_names[0])

                            component_to_check.info.spark_input_column_names.append(
                                other_component.info.spark_output_column_names[0])
                            component_to_check.model.setInputCols(
                                component_to_check.info.spark_input_column_names)

                            logger.info(
                                f'Setting input col columns for component {component_to_check.info.name} to {other_component.info.spark_output_column_names[0]} ')

        return pipe


    @staticmethod
    def check_and_fix_nlu_pipeline(pipe):
        """Check if the NLU pipeline is ready to transform data and return it.
        If all dependencies not satisfied, returns a new NLU pipeline where dependencies and sub-dependencies are satisfied.
        Checks and resolves in the following order :
        1. Get a reference list of input features missing for the current pipe
        2. Resolve the list of missing features by adding new  Annotators to pipe
        3. Add NER Converter if required (When there is a NER model)
        4. Fix order and output column names
        5.

        :param pipe:
        :return:
        """
        # main entry point for Model stacking withouth pretrained pipelines
        # requirements and provided features will be lists of lists

        #0. Clean old @AT storage ref from all columns
        logger.info('Cleaning old AT refs')
        pipe = PipeUtils.clean_AT_storage_refs(pipe)

        #1. Resolve dependencies, builds a DAG in reverse and satisfies dependencies with a Breadth-First-Search approach
        logger.info('Satisfying dependencies')
        pipe = PipelineQueryVerifier.satisfy_dependencies(pipe)

        #2. Enforce naming schema <col_name>@<storage_ref> for storage_ref consumers and producers and <entity@nlu_ref> and <ner@nlu_ref> for NER and NER-Converters
        # and add NER-IOB to NER-Pretty converters for every NER model that is not already feeding a NER converter
        pipe = PipeUtils.enforce_AT_schema_on_pipeline(pipe)

        #3. Validate naming of output columns is correct and no error will be thrown in spark
        logger.info('Fixing column names')
        pipe = PipelineQueryVerifier.check_and_fix_component_output_column_name_satisfaction(pipe)

        #4. Set on every NLP Annotator the output columns
        pipe = PipeUtils.enforce_NLU_columns_to_NLP_columns(pipe)

        #5. fix order
        logger.info('Optimizing pipe component order')
        pipe = PipelineQueryVerifier.check_and_fix_component_order(pipe)

        #6. Rename overlapping/duplicate leaf columns in the DAG
        logger.info('Renaming duplicates cols')
        pipe = PipeUtils.rename_duplicate_cols(pipe)

        #7. enfore again because trainable pipes might mutate pipe cols
        pipe = PipeUtils.enforce_NLU_columns_to_NLP_columns(pipe)

        logger.info('Done with pipe optimizing')

        return pipe




    @staticmethod
    def check_and_fix_component_order(pipe):
        '''
        This method takes care that the order of components is the correct in such a way,that the pipeline can be iteratively processed by spark NLP.
        Column Names will not be touched. DAG Task Sort basically.
        '''
        logger.info("Starting to optimize component order ")
        correct_order_component_pipeline = []
        all_components_orderd = False
        all_components = pipe.components
        provided_features = []
        update_last_type = False
        last_type_sorted = None
        trainable_updated = False
        while all_components_orderd == False:
            if update_last_type : last_type_sorted = None
            else : update_last_type = True
            for component in all_components:
                logger.info(f"Optimizing order for component {component.info.name}")
                # input_columns = ComponentUtils.clean_irrelevant_features(component.info.spark_input_column_names, False)
                input_columns = ComponentUtils.remove_storage_ref_from_features(
                    ComponentUtils.clean_irrelevant_features(component.info.spark_input_column_names,False))


                if last_type_sorted is None or component.info.type == last_type_sorted:
                    if set(input_columns).issubset(provided_features):
                        correct_order_component_pipeline.append(component)
                        if component in all_components: all_components.remove(component)
                        # for feature in component.info.spark_output_column_names: provided_features.append(feature)

                        # provided_features += ComponentUtils.clean_irrelevant_features(component.info.spark_output_column_names,False)
                        # TODO REMOVE STORAGE REF FROM PROVDED FEATURES???
                        provided_features += ComponentUtils.remove_storage_ref_from_features(
                            ComponentUtils.clean_irrelevant_features(component.info.spark_output_column_names,False))

                        last_type_sorted = component.info.type
                        update_last_type = False
                        break
            if len(all_components) == 0: all_components_orderd = True

            if not all_components_orderd  and len(all_components) <= 2 and pipe.has_trainable_components and not trainable_updated  and 'approach' in str(all_components[0].model).lower() and 'sentence_embeddings@' in all_components[0].info.inputs:
                # special case, if trainable then we feed embed consumers on the first sentence embed provider
                # 1. Find first sent embed provider
                # 2. substitute any 'sent_embed@' consumer inputs for the provider col
                for f in provided_features:
                    if 'sentence_embeddings' in f and not trainable_updated  :
                        all_components[0].info.spark_input_column_names.remove('sentence_embeddings@')
                        if 'sentence_embeddings@' in  all_components[0].info.inputs :  all_components[0].info.inputs.remove('sentence_embeddings@')
                        all_components[0].info.spark_input_column_names.append(f)
                        if f not in all_components[0].info.inputs :  all_components[0].info.inputs.append(f)
                        trainable_updated = True

            if not all_components_orderd and len(all_components) <= 2 and pipe.has_trainable_components and not trainable_updated  and 'approach' in str(all_components[0].model).lower() and 'word_embeddings@' in all_components[0].info.inputs:
                # special case, if trainable then we feed embed consumers on the first sentence embed provider
                # 1. Find first sent embed provider
                # 2. substitute any 'sent_embed@' consumer inputs for the provider col
                for f in provided_features:
                    if 'word_embeddings' in f and not trainable_updated  :
                        all_components[0].info.spark_input_column_names.remove('word_embeddings@')
                        if 'word_embeddings@' in  all_components[0].info.inputs :  all_components[0].info.inputs.remove('word_embeddings@')
                        all_components[0].info.spark_input_column_names.append(f)
                        if f not in all_components[0].info.inputs :  all_components[0].info.inputs.append(f)
                        trainable_updated = True


        pipe.components = correct_order_component_pipeline

        return pipe




    @staticmethod
    def log_resolution_status(provided_features_no_ref,required_features_no_ref,provided_features_ref,required_features_ref,is_trainable,conversion_candidates,missing_features_no_ref,missing_features_ref,):
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



    @staticmethod
    def is_storage_ref_match(embedding_consumer, embedding_provider,pipe):
        """Check for 2 components, if one provides the embeddings for the other. Makes sure that output_level matches up (chunk/sent/tok/embeds)"""
        consumer_AT_ref = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(embedding_consumer, 'input')
        provider_AT_rev = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(embedding_provider, 'output')
        consum_level    = ComponentUtils.extract_embed_level_identity(embedding_consumer, 'input')
        provide_level   = ComponentUtils.extract_embed_level_identity(embedding_provider, 'output')

        consumer_ref    = StorageRefUtils.extract_storage_ref(embedding_consumer)
        provider_rev    = StorageRefUtils.extract_storage_ref(embedding_provider)

        # input/output levels must match
        if consum_level != provide_level : return False

        # If storage ref dont match up, we must consult the storage_ref_2_embed mapping if it still maybe is a match, otherwise it is not.
        if consumer_ref == provider_rev  : return True

        # Embed Components have have been resolved via@ have a  nlu_resolution_ref_source will match up with the consumer ref if correct embedding.
        if hasattr(embedding_provider.info, 'nlu_ref'):
            if consumer_ref == StorageRefUtils.extract_storage_ref(embedding_provider.info.nlu_ref) : return True

        # If it is either  sentence_embedding_converter or chunk_embedding_converter then we gotta check what the storage ref of the inpot of those is.
        # If storage ref matches up, the providers output will match the consumer
        # if embedding_provider
        if embedding_provider.info.name in ["chunk_embedding_converter", 'sentence_embedding_converter']: # TODO FOR RESOLUTION
            nlu_ref, conv_prov_storage_ref = PipelineQueryVerifier.get_converters_provider_info(embedding_provider,pipe)


        return False





    @staticmethod
    def is_matching_level(embedding_consumer, embedding_provider):
        """Check for embedding consumer if input level matches up outputlevel of consumer
        """

    @staticmethod
    def get_converters_provider_info(embedding_provider,pipe):
        """For a component and a pipe, find storage_ref and """