import nlu
from nlu.pipeline import NLUPipeline

import logging
logger = logging.getLogger('nlu')

import inspect

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
    def is_untrained_model(component):
        '''
        Check for a given component if it is an embelishment of an traianble model.
        In this case we will ignore embeddings requirements further down the logic pipeline
        :param component: Component to check
        :return: True if it is trainable, False if not
        '''
        if 'is_untrained' in dict(inspect.getmembers(component.component_info)).keys() : return True
        return False
    @staticmethod
    def has_embeddings_requirement(component):
        '''
        Check for the input component, wether it depends on some embedding. Returns True if yes, otherwise False.

        :param component:  The component to check
        :return: True if the component needs some specifc embedding (i.e.glove, bert, elmo etc..). Otherwise returns False
        '''


        if type(component) == list or type(component) == set:
            for feature in component:
                if 'embed' in feature: return True
            return False
        else:
            for feature in component.component_info.inputs:
                if 'embed' in feature: return True
        return False

    @staticmethod
    def has_embeddings_provisions(component):
        '''
        Check for the input component, wether it depends on some embedding. Returns True if yes, otherwise False.
        :param component:  The component to check
        :return: True if the component needs some specifc embedding (i.e.glove, bert, elmo etc..). Otherwise returns False
        '''
        if type(component) == type(list) or type(component) == type(set):
            for feature in component:
                if 'embed' in feature: return True
            return False
        else:
            for feature in component.component_info.outputs:
                if 'embed' in feature: return True
        return False

    @staticmethod
    def clean_irrelevant_features(component_list):
        '''
        Remove irrelevant features from a list of component features
        :param component_list: list of features
        :return: list with only relevant feature names
        '''
        # remove irrelevant missing features for pretrained models
        if 'text' in component_list: component_list.remove('text')
        if 'raw_text' in component_list: component_list.remove('raw_text')
        if 'raw_texts' in component_list: component_list.remove('raw_texts')
        if 'label' in component_list: component_list.remove('label')
        if 'sentiment_label' in component_list: component_list.remove('sentiment_label')
        return component_list

    @staticmethod
    def get_missing_required_features(pipe: NLUPipeline):
        '''
        Takes in a NLUPipeline and returns a list of missing  feature types types which would cause the pipeline to crash if not added
        If it is some kind of model that uses embeddings, it will check the metadata for that model and return a string with moelName@spark_nlp_embedding_reference format
        '''
        logger.info('Resolving missing components')
        pipe_requirements = [['sentence',
                              'token']]  # default requirements so we can support all output levels. minimal extra comoputation effort. If we add CHUNK here, we will aalwayshave POS default
        pipe_provided_features = []
        pipe.has_trainable_components = False
        # pipe_types = [] # list of string identifiers
        for component in pipe.pipe_components:
            trainable = PipelineQueryVerifier.is_untrained_model(component)
            if trainable : pipe.has_trainable_components = True
            # 1. Get all feature provisions from the pipeline
            logger.info("Getting Missing Feature for component =%s", component.component_info.name)
            if not component.component_info.inputs == component.component_info.outputs:
                pipe_provided_features.append(
                    component.component_info.outputs)  # edge case for components that provide token and require token and similar cases.
            # 2. get all feature requirements for pipeline
            if PipelineQueryVerifier.has_embeddings_requirement(component) and not trainable:
                # special case for models with embedding requirements. we will modify the output string which then will be resolved by the default component resolver (which will get the correct embedding )
                if component.component_info.type == 'chunk_embeddings':
                    # there is no ref for Chunk embeddings, so we have a special case here and need to define a default value that will always be used for chunkers
                    sparknlp_embeddings_requirement_reference = 'glove'
                else:
                    sparknlp_embeddings_requirement_reference = component.model.extractParamMap()[
                        component.model.getParam('storageRef')]
                inputs_with_sparknlp_reference = []
                for feature in component.component_info.inputs:
                    if 'embed' in feature:
                        inputs_with_sparknlp_reference.append(feature + '@' + sparknlp_embeddings_requirement_reference)
                    else:
                        inputs_with_sparknlp_reference.append(feature)
                pipe_requirements.append(inputs_with_sparknlp_reference)
            else:
                pipe_requirements.append(component.component_info.inputs)

        # 3. Some components have "word_embeddings" als input configured, but no actual wordembedding has "word_embedding" as output configured.
        # Thus we must check in a different way here first if embeddings are provided and if they are there we have to remove them form the requirements list

        # 4. get missing requirements, by substracting provided from requirements
        # Flatten lists, make them to sets and get missing components by substracting them.
        flat_requirements = set(item for sublist in pipe_requirements for item in sublist)
        flat_provisions = set(item for sublist in pipe_provided_features for item in sublist)
        # rmv spark identifier from provision
        flat_requirements_no_ref = set(item.split('@')[0] if '@' in item else item for item in flat_requirements)

        # see what is missing, with identifier removed
        missing_components = PipelineQueryVerifier.clean_irrelevant_features(flat_requirements_no_ref - flat_provisions)
        logger.info("Required columns no ref flat =%s", flat_requirements_no_ref)
        logger.info("Provided columns flat =%s", flat_provisions)
        logger.info("Missing columns no ref flat =%s", missing_components)
        # since embeds are missing, we add embed with reference back
        if PipelineQueryVerifier.has_embeddings_requirement(missing_components) and not trainable:
            missing_components = PipelineQueryVerifier.clean_irrelevant_features(flat_requirements - flat_provisions)

        if len(missing_components) == 0:
            logger.info('No more components missing!')
            return []
        else:
            # we must recaclulate the difference, because we reoved the spark nlp reference previously for our set operation. Since it was not 0, we ad the Spark NLP rererence back
            logger.info('Components missing=%s', str(missing_components))
            return missing_components

    @staticmethod
    def add_ner_converter_if_required(pipe: NLUPipeline):
        '''
        This method loops over every component in the pipeline and check if any of them outputs an NER type column.
        If NER exists in the pipeline, then this method checks if NER converter is already in pipeline.
        If NER exists and NER converter is NOT in pipeline, NER converter will be added to pipeline.
        :param pipe: The pipeline we wish to configure ner_converter dependency for
        :return: pipeline with NER configured
        '''

        ner_converter_exists = False
        ner_exists = False

        for component in pipe.pipe_components:
            if 'ner' in component.component_info.outputs: ner_exists = True
            if 'entities' in component.component_info.outputs: ner_converter_exists = True

        if ner_converter_exists == True:
            logger.info('NER converter already in pipeline')
            return pipe

        if not ner_converter_exists and ner_exists:
            logger.info('Adding NER Converter to pipeline')
            pipe.add(nlu.get_default_component_of_type(('ner_converter')))

        return pipe

    @staticmethod
    def check_and_fix_nlu_pipeline(pipe: NLUPipeline):
        # main entry point for Model stacking withouth pretrained pipelines
        # requirements and provided features will be lists of lists
        all_features_provided = False
        while all_features_provided == False:
            # After new components have been added, we must loop agan and check for the new components if requriements are met
            # OR we implement a function caled "Add components with requirements". That one needs to know though, which requirements are already met ...

            # Find missing components
            missing_components = PipelineQueryVerifier.get_missing_required_features(pipe)
            if len(missing_components) == 0: break  # Now all features are provided

            components_to_add = []
            # Create missing components
            for missing_component in missing_components:
                if 'embedding' in missing_component or 'token' in missing_component: components_to_add.append(nlu.get_default_component_of_type(missing_component,language= pipe.lang))
                else: components_to_add.append(nlu.get_default_component_of_type(missing_component))

            logger.info('Resolved for missing components the following NLU components : %s', str(components_to_add))

            # Add missing components and validate order of components is correct
            for new_component in components_to_add:
                pipe.add(new_component)
                logger.info('adding %s=', new_component.component_info.name)

            # 3 Add NER converter if NER component is in pipeline : (This is a bit ineficcent but it is most stable)
            pipe = PipelineQueryVerifier.add_ner_converter_if_required(pipe)

        logger.info('Fixing column names')
        #  Validate naming of output columns is correct and no error will be thrown in spark
        pipe = PipelineQueryVerifier.check_and_fix_component_output_column_name_satisfaction(pipe)

        # 4.  fix order
        logger.info('Optimizing pipe component order')
        pipe = PipelineQueryVerifier.check_and_fix_component_order(pipe)

        # 5. Check if output column names overlap, if yes, fix
        # pipe = PipelineQueryVerifier.check_and_fix_component_order(pipe)
        # 6.  Download all file depenencies like train files or  dictionaries
        logger.info('Done with pipe optimizing')

        return pipe

    @staticmethod
    def check_and_fix_component_output_column_name_satisfaction(pipe: NLUPipeline):
        '''
        This function verifies that every input and output column name of a component is satisfied.
        If some output names are missing, it will be added by this methods.
        Usually classifiers need to change their input column name, so that it matches one of the previous embeddings because they have dynamic output names
        This function peforms the following steps :
        1. For each component we veryify that all input column names are satisfied  by checking all other components output names
        2. When a input column is missing we do the following :
        2.1 Figure out the type of the missing input column. The name of the missing column should be equal to the type
        2.2 Check if there is already a component in the pipe, which provides this input (It should)
        2.3. When the providing component is found, update its output name, or update the original coponents input name
        :return: NLU pipeline where the output and input column names of the models have been adjusted to each other
        '''


        for component_to_check in pipe.pipe_components:
            input_columns = set(component_to_check.component_info.spark_input_column_names)
            logger.info(f'Checking for component {component_to_check.component_info.name} wether inputs {input_columns} is satisfied by another component in the pipe ',)
            for other_component in pipe.pipe_components:
                if component_to_check.component_info.name == other_component.component_info.name: continue
                output_columns = set(other_component.component_info.spark_output_column_names)
                input_columns -= output_columns

            input_columns = PipelineQueryVerifier.clean_irrelevant_features(input_columns)

            if len(input_columns) != 0 and not pipe.has_trainable_components:  # fix missing column name
                logger.info(f"Fixing bad input col for C={component_to_check} untrainable pipe")
                for missing_column in input_columns:
                    for other_component in pipe.pipe_components:
                        if component_to_check.component_info.name == other_component.component_info.name: continue
                        if other_component.component_info.type == missing_column:
                            # We update the output name for the component which provides our feature
                            other_component.component_info.spark_output_column_names = [missing_column]
                            logger.info('Setting output columns for component %s to %s ',
                                        other_component.component_info.name, missing_column)
                            other_component.model.setOutputCol(missing_column)

            elif len(input_columns) != 0 and  pipe.has_trainable_components:  # fix missing column name
                logger.info(f"Fixing bad input col for C={component_to_check} trainable pipe")

                # for trainable components, we change their input columns and leave other components outputs unchanged
                for missing_column in input_columns:
                    for other_component in pipe.pipe_components:
                        if component_to_check.component_info.name == other_component.component_info.name: continue
                        if other_component.component_info.type == missing_column:
                            # We update the input col name for the componenet that has missing cols
                            component_to_check.component_info.spark_input_column_names.remove(missing_column)
                            # component_to_check.component_info.inputs.remove(missing_column)
                            # component_to_check.component_info.inputs.remove(missing_column)
                            # component_to_check.component_info.inputs.append(other_component.component_info.spark_output_column_names[0])

                            component_to_check.component_info.spark_input_column_names.append(other_component.component_info.spark_output_column_names[0])
                            component_to_check.model.setInputCols(component_to_check.component_info.spark_input_column_names)

                            logger.info(f'Setting input col columns for component {component_to_check.component_info.name} to {other_component.component_info.spark_output_column_names[0]} ')



        return pipe

    @staticmethod
    def check_and_fix_component_output_column_name_overlap(pipe: NLUPipeline):
        '''
        #todo use?
        This method enforces that every component has a unique output column name.
        Especially for classifiers or bert_embeddings this issue might occur,


        1. For each component we veryify that all input column names are satisfied  by checking all other components output names
        2. When a input column is missing we do the following :
        2.1 Figure out the type of the missing input column. The name of the missing column should be equal to the type
        2.2 Check if there is already a component in the pipe, which provides this input (It should)
        2.3. When the providing component is found, update its output name, or update the original coponents input name
        :return: NLU pipeline where the output and input column names of the models have been adjusted to each other
        '''

        all_names_provided = False

        for component_to_check in pipe.pipe_components:
            all_names_provided_for_component = False
            input_columns = set(component_to_check.component_info.spark_input_column_names)
            logger.info('Checking for component %s wether input %s is satisfied by another component in the pipe ',
                        component_to_check.component_info.name, input_columns)
            for other_component in pipe.pipe_components:
                if component_to_check.component_info.name == other_component.component_info.name: continue
                output_columns = set(other_component.component_info.spark_output_column_names)
                input_columns -= output_columns  # set substraction

            input_columns = PipelineQueryVerifier.clean_irrelevant_features(input_columns)

            if len(input_columns) != 0:  # fix missing column name
                for missing_column in input_columns:
                    for other_component in pipe.pipe_components:
                        if component_to_check.component_info.name == other_component.component_info.name: continue
                        if other_component.component_info.type == missing_column:
                            # resolve which setter to use ...
                            # We update the output name for the component which provides our feature
                            other_component.component_info.spark_output_column_names = [missing_column]
                            logger.info('Setting output columns for component %s to %s ',
                                        other_component.component_info.name, missing_column)
                            other_component.model.setOutputCol(missing_column)

        return pipe

    @staticmethod
    def check_and_fix_component_order(pipe: NLUPipeline):
        '''
        This method takes care that the order of components is the correct in such a way,
        that the pipeline can be iteratively processed by spark NLP.
        If output_level == Document, then sentence embeddings will be fed on Document col and classifiers recieve doc_embeds/doc_raw column, depending on if the classifier works with or withouth embeddings
        If output_level == sentence, then sentence embeddings will be fed on sentence col and classifiers recieve sentence_embeds/sentence_raw column, depending on if the classifier works with or withouth embeddings. IF sentence detector is missing, one will be added.

        '''
        logger.info("Starting to optimize component order ")
        correct_order_component_pipeline = []
        all_components_orderd = False
        all_components = pipe.pipe_components
        provided_features = []
        while all_components_orderd == False:
            for component in all_components:
                logger.info("Optimizing order for component %s", component.component_info.name)
                input_columns = PipelineQueryVerifier.clean_irrelevant_features(component.component_info.inputs)
                if set(input_columns).issubset(provided_features):
                    correct_order_component_pipeline.append(component)
                    if component in all_components: all_components.remove(component)
                    for feature in component.component_info.outputs: provided_features.append(feature)
            if len(all_components) == 0: all_components_orderd = True

        pipe.pipe_components = correct_order_component_pipeline

        return pipe

    @staticmethod
    def configure_component_output_levels(pipe: NLUPipeline):
        '''
        This method configures sentenceEmbeddings and Classifier components to output at a specific level
        This method is called the first time .predit() is called and every time the output_level changed
        If output_level == Document, then sentence embeddings will be fed on Document col and classifiers recieve doc_embeds/doc_raw column, depending on if the classifier works with or withouth embeddings
        If output_level == sentence, then sentence embeddings will be fed on sentence col and classifiers recieve sentence_embeds/sentence_raw column, depending on if the classifier works with or withouth embeddings. IF sentence detector is missing, one will be added.

        '''
        if pipe.output_level == 'sentence':
            return PipelineQueryVerifier.configure_component_output_levels_to_sentence(pipe)
        elif pipe.output_level == 'document':
            return PipelineQueryVerifier.configure_component_output_levels_to_document(pipe)

    @staticmethod
    def configure_component_output_levels_to_sentence(pipe: NLUPipeline):
        '''
        Configure pipe compunoents to output level document
        :param pipe: pipe to be configured
        :return: configured pipe
        '''
        for c in pipe.pipe_components:
            if 'token' in c.component_info.spark_output_column_names: continue
            # if 'document' in c.component_info.inputs and 'sentence' not in c.component_info.inputs  :
            if 'document' in c.component_info.inputs and 'sentence' not in c.component_info.inputs and 'sentence' not in c.component_info.outputs:
                logger.info(f"Configuring C={c.component_info.name}  of Type={type(c.model)}")
                c.component_info.inputs.remove('document')
                c.component_info.inputs.append('sentence')
                # c.component_info.spark_input_column_names.remove('document')
                # c.component_info.spark_input_column_names.append('sentence')
                c.model.setInputCols(c.component_info.spark_input_column_names)

            if 'document' in c.component_info.spark_input_column_names and 'sentence' not in c.component_info.spark_input_column_names and 'sentence' not in c.component_info.spark_output_column_names:
                c.component_info.spark_input_column_names.remove('document')
                c.component_info.spark_input_column_names.append('sentence')
                if c.component_info.type =='sentence_embeddings' :
                    c.component_info.output_level='sentence'

        return pipe

    @staticmethod
    def configure_component_output_levels_to_document(pipe: NLUPipeline):
        '''
        Configure pipe compunoents to output level document
        :param pipe: pipe to be configured
        :return: configured pipe
        '''
        logger.info('Configuring components to document level')
        # Every sentenceEmbedding can work on Dcument col
        # This works on the assuption that EVERY annotator that works on sentence col, can also work on document col. Douple Tripple verify later
        # here we could change the col name to doc_embedding potentially
        # 1. Configure every Annotator/Classifier that works on sentences to take in Document instead of sentence
        #  Note: This takes care of changing Sentence_embeddings to Document embeddings, since embedder runs on doc then.
        for c in pipe.pipe_components:
            if 'token' in c.component_info.spark_output_column_names: continue
            if 'sentence' in c.component_info.inputs and 'document' not in c.component_info.inputs:
                logger.info(f"Configuring C={c.component_info.name}  of Type={type(c.model)} input to document level")
                c.component_info.inputs.remove('sentence')
                c.component_info.inputs.append('document')

            if 'sentence' in c.component_info.spark_input_column_names and 'document' not in c.component_info.spark_input_column_names:
                # if 'sentence' in c.component_info.spark_input_column_names : c.component_info.spark_input_column_names.remove('sentence')
                c.component_info.spark_input_column_names.remove('sentence')
                c.component_info.spark_input_column_names.append('document')
                c.model.setInputCols(c.component_info.spark_input_column_names)

            if c.component_info.type =='sentence_embeddings' : #convert sentence embeds to doc
                c.component_info.output_level='document'

        return pipe

    @staticmethod
    def configure_output_to_most_recent(pipe: NLUPipeline):
        '''
        For annotators that feed on tokens, there are often multiple options of tokens they could feed on, i,e, spell/norm/lemma/stemm
        This method enforces that each annotator that takes in tokens will be fed the MOST RECENTLY ADDED token, unless specified in the NLU_load parameter otherwise

        :param pipe:
        :return:
        '''
        pass

    @staticmethod
    def has_sentence_emebddings_col(component):
        '''
        Check for a given component if it uses sentence embedding col
        :param component: component to check
        :return: True if uses raw sentences, False if not
        '''
        for inp in component.component_info.inputs:
            if inp == 'sentence_embeddings': return True
        return False

    @staticmethod
    def is_using_token_level_inputs(component):
        '''
        Check for a given component if it uses Token level input
        I.e. Lemma/stem/token/ and return the col name if so
        :param component: component to check
        :return: True if uses raw sentences, False if not
        '''
        token_inputs = ['token', 'lemma', 'stem', 'spell', '']
        for inp in component.component_info.inputs:
            if inp == 'sentence_embeddings': return True
        return False
