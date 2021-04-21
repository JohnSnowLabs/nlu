import inspect
import logging
logger = logging.getLogger('nlu')

from nlu.pipe.pipe_components import SparkNLUComponent
from nlu.pipe.utils.component_utils import ComponentUtils
from nlu.pipe.utils.storage_ref_utils import StorageRefUtils
# from nlu.pipe.pipeline import NLUPipeline cannot import or we have circular dep
"""Pipe Level logic oprations and utils"""
class PipeUtils():
    @staticmethod
    def is_trainable_pipe(pipe):
        '''Check if pipe is trainable'''
        for c in pipe.components:
            if ComponentUtils.is_untrained_model(c):return True
        return False



    @staticmethod
    def enforece_AT_embedding_provider_output_col_name_schema_for_list_of_components  (pipe_list):
        """For every embedding provider, enforce that their output col is named <output_level>@storage_ref for output_levels word,chunk,sentence aka document , i.e. word_embed@elmo or sentence_embed@elmo etc.."""
        for c in pipe_list:
            if ComponentUtils.is_embedding_provider(c):
                level_AT_ref = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(c, 'output')
                c.info.outputs = [level_AT_ref]
                c.info.spark_output_column_names = [level_AT_ref]
                c.model.setOutputCol(level_AT_ref[0])
        return pipe_list


    @staticmethod
    def enforce_AT_schema_on_pipeline(pipe):
        """Enforces the AT naming schema on all column names and add missing NER converters"""
        return PipeUtils.enforce_AT_schema_on_NER_processors_and_add_missing_NER_converters(PipeUtils.enforce_AT_schema_on_embedding_processors(pipe))

    @staticmethod
    def enforce_AT_schema_on_NER_processors_and_add_missing_NER_converters(pipe):
        """For every NER provider and consumer, enforce that their output col is named <output_level>@storage_ref for output_levels word,chunk,sentence aka document , i.e. word_embed@elmo or sentence_embed@elmo etc..
        We also add NER converters for every NER model that no Converter converting it's inputs
        In addition, returns the pipeline with missing NER converters added, for every NER model.
        The converters transform the IOB schema in a merged and more usable form for downstream tasks
        1. Find a NER model in pipe
        2. Find a NER converter feeding from it, if there is None, create one.
        3. Generate name with Identifier  <ner-iob>@<nlu_ref_identifier>  and <entities>@<nlu_ref_identifier>
        3.1 Update NER Models    output to <ner-iob>@<nlu_ref_identifier>
        3.2 Update NER Converter input  to <ner-iob>@<nlu_ref_identifier>
        3.3 Update NER Converter output to <entities>@<nlu_ref_identifier>
        4. Update every Component that feeds from the NER converter (i.e. Resolver etc..)
        """
        from nlu import Util
        new_converters = []
        for c in pipe.components:
            if ComponentUtils.is_NER_provider(c):
                output_NER_col = ComponentUtils.extract_NER_col(c,'output')
                converter_to_update = None
                # if '@' not in output_NER_col:
                for other_c in pipe.components:
                    if output_NER_col in other_c.info.inputs and ComponentUtils.is_NER_converter(other_c):
                        converter_to_update = other_c

                ner_identifier = ComponentUtils.get_nlu_ref_identifier(c)
                if converter_to_update is  None :
                    if c.info.license == 'healthcare': converter_to_update  = Util("ner_to_chunk_converter_licensed",is_licensed=True )
                    else : converter_to_update  = Util("ner_to_chunk_converter")
                    new_converters.append(converter_to_update)

                converter_to_update.info.nlu_ref = f'ner_converter@{ner_identifier}'

                # 3. generate new col names
                new_NER_AT_ref = output_NER_col
                if '@' not in output_NER_col: new_NER_AT_ref = output_NER_col + '@' + ner_identifier
                new_NER_converter_AT_ref = 'entities' + '@' + ner_identifier

                # 3.1 upate NER model outputs
                c.info.outputs = [new_NER_AT_ref]
                c.info.spark_output_column_names = [new_NER_AT_ref]
                c.model.setOutputCol(new_NER_AT_ref)

                #3.2 update converter inputs
                old_ner_input_col = ComponentUtils.extract_NER_converter_col(converter_to_update, 'input')
                converter_to_update.info.inputs.remove(old_ner_input_col)
                converter_to_update.info.spark_input_column_names.remove(old_ner_input_col)
                converter_to_update.info.inputs.append(new_NER_AT_ref)
                converter_to_update.info.spark_input_column_names.append(new_NER_AT_ref)
                converter_to_update.model.setInputCols(converter_to_update.info.inputs)

                #3.3 update converter outputs
                converter_to_update.info.outputs = [new_NER_converter_AT_ref]
                converter_to_update.info.spark_output_column_names = [new_NER_converter_AT_ref]
                converter_to_update.model.setOutputCol(new_NER_converter_AT_ref)

                ## todo improve, this causes the first ner producer to feed to all ner-cosnuners. All other ner-producers will be ignored by ner-consumers,w ithouth special syntax or manual configs
                ##4. Update all NER consumers input columns
                for conversion_consumer in pipe.components :
                    if 'entities' in conversion_consumer.info.inputs:
                        conversion_consumer.info.inputs.remove('entities')
                        conversion_consumer.info.spark_input_column_names.remove('entities')
                        conversion_consumer.info.inputs.append(new_NER_converter_AT_ref)
                        conversion_consumer.info.spark_input_column_names.append(new_NER_converter_AT_ref)

        # Add new converters to pipe
        for conv in new_converters:
            if conv.info.license == 'healthcare':
                pipe.add(conv,  name_to_add = f'chunk_converter_licensed@{conv.info.outputs[0].split("@")[0]}')
            else :
                pipe.add(conv,  name_to_add = f'chunk_converter@{conv.info.outputs[0].split("@")[0]}')
        return pipe

    @staticmethod
    def enforce_AT_schema_on_embedding_processors  (pipe):
        """For every embedding provider and consumer, enforce that their output col is named <output_level>@storage_ref for output_levels word,chunk,sentence aka document , i.e. word_embed@elmo or sentence_embed@elmo etc.."""
        for c in pipe.components:
            if ComponentUtils.is_embedding_provider(c):
                if '@' not in c.info.outputs[0]:
                    new_embed_AT_ref = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(c, 'output')
                    c.info.outputs = [new_embed_AT_ref]
                    c.info.spark_output_column_names = [new_embed_AT_ref]
                    # c.model.setOutputCol(new_embed_AT_ref[0]) # why [0] here?? bug!
                    c.model.setOutputCol(new_embed_AT_ref)

            if ComponentUtils.is_embedding_consumer(c):
                input_embed_col = ComponentUtils.extract_embed_col(c)
                if '@' not in input_embed_col:
                    # storage_ref = StorageRefUtils.extract_storage_ref(c)
                    # new_embed_col_with_AT_notation = input_embed_col+"@"+storage_ref
                    new_embed_AT_ref = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(c, 'input')
                    c.info.inputs.remove(input_embed_col)
                    c.info.inputs.append(new_embed_AT_ref)
                    c.info.spark_input_column_names.remove(input_embed_col)
                    c.info.spark_input_column_names.append(new_embed_AT_ref)
                    c.model.setInputCols(c.info.inputs)






        return pipe


    @staticmethod
    def enforce_NLU_columns_to_NLP_columns  (pipe):
        """for every component, set its inputs and outputs to the ones configured on the NLU component."""
        for c in pipe.components:
            if c.info.name == 'document_assembler' : continue
            c.model.setOutputCol(c.info.outputs[0])
            c.model.setInputCols(c.info.inputs)
            c.info.spark_input_column_names = c.info.inputs
            c.info.spark_output_column_names = c.info.outputs

        return pipe
    @staticmethod
    def is_converter_component_resolution_reference(reference:str)-> bool:
        if 'chunk_emb' in reference : return True
        if 'chunk_emb' in reference : return True

    @staticmethod
    def configure_component_output_levels_to_sentence(pipe):
        '''
        Configure pipe compunoents to output level document
        :param pipe: pipe to be configured
        :return: configured pipe
        '''
        for c in pipe.components:
            if 'token' in c.info.spark_output_column_names: continue
            # if 'document' in c.component_info.inputs and 'sentence' not in c.component_info.inputs  :
            if 'document' in c.info.inputs and 'sentence' not in c.info.inputs and 'sentence' not in c.info.outputs:
                logger.info(f"Configuring C={c.info.name}  of Type={type(c.model)}")
                c.info.inputs.remove('document')
                c.info.inputs.append('sentence')
                # c.component_info.spark_input_column_names.remove('document')
                # c.component_info.spark_input_column_names.append('sentence')
                c.model.setInputCols(c.info.spark_input_column_names)

            if 'document' in c.info.spark_input_column_names and 'sentence' not in c.info.spark_input_column_names and 'sentence' not in c.info.spark_output_column_names:
                c.info.spark_input_column_names.remove('document')
                c.info.spark_input_column_names.append('sentence')
                if c.info.type == 'sentence_embeddings':
                    c.info.output_level = 'sentence'

        return pipe

    @staticmethod
    def configure_component_output_levels_to_document(pipe):
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
        for c in pipe.components:
            if 'token' in c.info.spark_output_column_names: continue
            if 'sentence' in c.info.inputs and 'document' not in c.info.inputs:
                logger.info(f"Configuring C={c.info.name}  of Type={type(c.model)} input to document level")
                c.info.inputs.remove('sentence')
                c.info.inputs.append('document')

            if 'sentence' in c.info.spark_input_column_names and 'document' not in c.info.spark_input_column_names:
                # if 'sentence' in c.component_info.spark_input_column_names : c.component_info.spark_input_column_names.remove('sentence')
                c.info.spark_input_column_names.remove('sentence')
                c.info.spark_input_column_names.append('document')
                c.model.setInputCols(c.info.spark_input_column_names)

            if c.info.type == 'sentence_embeddings':  # convert sentence embeds to doc
                c.info.output_level = 'document'

        return pipe

    @staticmethod
    def configure_component_output_levels(pipe):
        '''
        This method configures sentenceEmbeddings and Classifier components to output at a specific level
        This method is called the first time .predit() is called and every time the output_level changed
        If output_level == Document, then sentence embeddings will be fed on Document col and classifiers recieve doc_embeds/doc_raw column, depending on if the classifier works with or withouth embeddings
        If output_level == sentence, then sentence embeddings will be fed on sentence col and classifiers recieve sentence_embeds/sentence_raw column, depending on if the classifier works with or withouth embeddings. IF sentence detector is missing, one will be added.

        '''
        if pipe.output_level == 'sentence':
            return PipeUtils.configure_component_output_levels_to_sentence(pipe)
        elif pipe.output_level == 'document':
            return PipeUtils.configure_component_output_levels_to_document(pipe)

    @staticmethod
    def check_if_component_is_in_pipe(pipe, component_name_to_check, check_strong=True):
        """Check if a component with a given name is already in a pipe """
        for c in pipe.components :
            if   check_strong and component_name_to_check == c.info.name : return True
            elif not check_strong and component_name_to_check in c.info.name : return True
        return False

    @staticmethod
    def check_if_there_component_with_col_in_components(component_list, features, except_component):
        """For a given list of features and a list of components, see if there are components taht provide this feature
        If yes, True, otherwise False
        """
        for c in component_list : 
            if c.info.outputs[0] != except_component.info.outputs[0] :
                for f in ComponentUtils.clean_irrelevant_features(c.info.spark_output_column_names, True):
                    if f in features : return True

        return False

    @staticmethod
    def is_leaf_node(c,pipe)-> bool:
        """Check if a component is a leaf in the DAG.
        We verify by checking if any other_c is feeding from c.
        If yes, it is not a leaf. If nobody feeds from c, it's a leaf.
        """
        inputs = c.info.inputs
        for other_c in pipe.components:
            if c is not other_c :
                for f in other_c.info.inputs :1


        return False



    @staticmethod
    def subsitute_leaf_output_names(pipe):
        """Change all output column names of leaves to something nicer, if they not already
        use AT notation"""

        for c in pipe.components:
            if PipeUtils.is_leaf_node(c,pipe) and not ComponentUtils.has_AT_notation():
                # update name
                1

        return pipe



    @staticmethod
    def clean_AT_storage_refs(pipe):
        """Removes AT notation from all columns. Useful to reset pipe back to default state"""

        for c in pipe.components:
            c.info.inputs  = [f.split('@')[0] for f in c.info.inputs]
            c.info.outputs = [f.split('@')[0] for f in c.info.outputs]

            c.info.spark_input_column_names  = [f.split('@')[0] for f in c.info.spark_input_column_names]
            c.info.spark_output_column_names = [f.split('@')[0] for f in c.info.spark_output_column_names]
        return pipe