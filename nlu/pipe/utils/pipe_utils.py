from sparknlp.annotator import *
import inspect
import logging
from nlu.pipe.nlu_component import NluComponent
from nlu.pipe.utils.resolution.storage_ref_utils import StorageRefUtils
from nlu.universe.logic_universes import NLP_LEVELS, AnnoTypes
from nlu import Licenses
from nlu.universe.feature_node_ids import NLP_NODE_IDS, NLP_HC_NODE_IDS
from nlu.universe.feature_universes import NLP_FEATURES
from nlu.universe.component_universes import ComponentMap

logger = logging.getLogger('nlu')
from nlu.pipe.utils.component_utils import ComponentUtils
from typing import List
from nlu.universe.annotator_class_universe import AnnoClassRef


class PipeUtils:
    """Pipe Level logic operations and utils"""

    @staticmethod
    def set_column_values_on_components_from_pretrained_pipe(component_list: List[NluComponent], nlp_ref, lang, path):
        """Since output/input cols cannot be fetched from Annotators via get input/output col reliably, we must check
        annotator data to find them Expects a list of NLU Component objects which all stem from the same pipeline
        defined by nlp_ref
        """
        import os
        import glob
        import json
        if path:
            pipe_path = path
        else:
            pipe_path = os.path.expanduser('~') + '/cache_pretrained/' + f'{nlp_ref}_{lang}'
            # WE do not need to check for Spark Version, since cols should match accors versions
            pipe_path = glob.glob(f'{pipe_path}*')[0]
            if not os.path.exists(pipe_path): raise FileNotFoundError(
                f"Could not find downloaded Pipeline at path={pipe_path}")

        # Find HDD location of component_list and read out input/output cols
        digits_num = len(str(len(component_list)))
        digit_str = '0' * digits_num
        digit_cur = 0
        for c in component_list:
            c_metadata_path = f'{pipe_path}/stages/{digit_str}_*/metadata/part-00000'
            c_metadata_path = glob.glob(f'{c_metadata_path}*')[0]
            with open(c_metadata_path, "r") as f:
                data = json.load(f)
                if 'inputCols' in data['paramMap'].keys():
                    inp = data['paramMap']['inputCols']
                    c.model.setInputCols(inp)
                else:
                    inp = data['paramMap']['inputCol']
                    c.model.setInputCol(inp)
                out = data['paramMap']['outputCol']
                c.spark_input_column_names = inp if isinstance(inp, List) else [inp]
                c.spark_output_column_names = [out]
                c.model.setOutputCol(out)
            digit_cur += 1
            digit_str = str(digit_cur)
            while len(digit_str) < digits_num:
                digit_str = '0' + digit_str
        return component_list

    @staticmethod
    def is_trainable_pipe(pipe):
        """Check if component_list is trainable"""
        for c in pipe.components:
            if c.trainable: return True
        return False

    @staticmethod
    def enforece_AT_embedding_provider_output_col_name_schema_for_list_of_components(pipe_list):
        """For every embedding provider, enforce that their output col is named <output_level>@storage_ref for
        output_levels word,chunk,sentence aka document , TODO update the classifier models swell i.e.
        word_embed@elmo or sentence_embed@elmo etc. """
        for c in pipe_list:
            if ComponentUtils.is_embedding_provider(c):
                level_AT_ref = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(c, 'output')
                c.out_types = [level_AT_ref]
                c.info.spark_output_column_names = [level_AT_ref]
                c.model.setOutputCol(level_AT_ref[0])
        return pipe_list

    @staticmethod
    def enforce_AT_schema_on_pipeline_and_add_NER_converter(pipe):
        """Enforces the AT naming schema on all column names and add missing NER converters"""
        return PipeUtils.enforce_AT_schema_on_NER_processors_and_add_missing_NER_converters(
            PipeUtils.enforce_AT_schema_on_embedding_processors(pipe))

    @staticmethod
    def enforce_AT_schema_on_NER_processors_and_add_missing_NER_converters(pipe):
        """For every NER provider and consumer, enforce that their output col is named <output_level>@storage_ref for
        output_levels word,chunk,sentence aka document , i.e. word_embed@elmo or sentence_embed@elmo etc. We also
        add NER converters for every NER model that no Converter converting its inputs In addition, returns the
        pipeline with missing NER converters added, for every NER model. The converters transform the IOB schema in a
        merged and more usable form for downstream tasks 1. Find a NER model in component_list 2. Find a NER
        converter feeding from it, if there is None, create one. 3. Generate name with Identifier
        <ner-iob>@<nlu_ref_identifier>  and <entities>@<nlu_ref_identifier> 3.1 Update NER Models    output to
        <ner-iob>@<nlu_ref_identifier> 3.2 Update NER Converter input  to <ner-iob>@<nlu_ref_identifier> 3.3 Update
        NER Converter output to <entities>@<nlu_ref_identifier> 4. Update every Component that feeds from the NER
        converter (i.e. Resolver etc.)

        includes TOKEN-CLASSIFIER-TRANSFORMER models which usually output NER format
        """
        from nlu.components.util import Util
        new_converters = []
        for c in pipe.components:
            if c.loaded_from_pretrained_pipe:
                # Leave pretrained component_list models untouched
                new_converters.append(c)
                continue

            # TRANSFORMER_TOKEN_CLASSIFIER might be a NER provider. Regardless, No ner-Conversion will be performed
            # because it will not return NER IOB
            if ComponentUtils.is_NER_provider(c) and c.type != AnnoTypes.TRANSFORMER_TOKEN_CLASSIFIER:

                output_NER_col = ComponentUtils.extract_NER_col(c, 'output')
                converter_to_update = None
                for other_c in pipe.components:
                    if output_NER_col in other_c.spark_input_column_names and ComponentUtils.is_NER_converter(other_c):
                        converter_to_update = other_c

                ner_identifier = ComponentUtils.get_nlu_ref_identifier(c)
                if converter_to_update is None:
                    if c.license == Licenses.hc:
                        # TODO SET METADATA FIELDS HERE ON ANNO!!
                        converter_to_update = ComponentMap.hc_components[NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL]
                        converter_to_update.set_metadata(converter_to_update.get_default_model(),
                                                         NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL,
                                                         NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL,
                                                         'xx', False, Licenses.hc)
                    else:
                        # TODO SET METADATA FIELDS HERE ON ANNO!!
                        converter_to_update = ComponentMap.os_components[NLP_NODE_IDS.NER_CONVERTER]
                        converter_to_update.set_metadata(converter_to_update.get_default_model(),
                                                         NLP_NODE_IDS.NER_CONVERTER, NLP_NODE_IDS.NER_CONVERTER,
                                                         'xx', False, Licenses.open_source)

                    new_converters.append(converter_to_update)
                converter_to_update.nlu_ref = f'ner_converter.{c.nlu_ref}'

                # 3. generate new col names
                new_NER_AT_ref = output_NER_col
                if '@' not in output_NER_col: new_NER_AT_ref = output_NER_col + '@' + ner_identifier
                new_NER_converter_AT_ref = 'entities' + '@' + ner_identifier

                # 3.1 upate NER model outputs
                c.spark_output_column_names = [new_NER_AT_ref]
                c.model.setOutputCol(new_NER_AT_ref)

                # 3.2 update converter inputs
                old_ner_input_col = ComponentUtils.extract_NER_converter_col(converter_to_update, 'input')
                if old_ner_input_col in converter_to_update.spark_input_column_names:
                    converter_to_update.spark_input_column_names.remove(old_ner_input_col)
                else:
                    converter_to_update.spark_input_column_names.pop()

                # if old_ner_input_col in converter_to_update.spark_input_column_names:
                #     converter_to_update.spark_input_column_names.remove(old_ner_input_col)
                # else:
                #     converter_to_update.spark_input_column_names.pop()
                converter_to_update.spark_input_column_names.append(new_NER_AT_ref)
                converter_to_update.model.setInputCols(converter_to_update.spark_input_column_names)

                # 3.3 update converter outputs
                converter_to_update.spark_output_column_names = [new_NER_converter_AT_ref]
                converter_to_update.model.setOutputCol(new_NER_converter_AT_ref)

                ## todo improve, this causes the first ner producer to feed to all ner-cosnuners. All other ner-producers will be ignored by ner-consumers,w ithouth special syntax or manual configs --> Chunk merger
                ##4. Update all NER consumers input columns, i.e. Resolver, Relation, etc..
                for conversion_consumer in pipe.components:
                    if NLP_FEATURES.NAMED_ENTITY_CONVERTED in conversion_consumer.in_types:
                        conversion_consumer.spark_input_column_names.remove(NLP_FEATURES.NAMED_ENTITY_CONVERTED)
                        conversion_consumer.spark_input_column_names.append(new_NER_converter_AT_ref)

        # Add new converters to component_list
        for conv in new_converters:
            if conv.license == Licenses.hc:
                pipe.add(conv,
                         name_to_add=f'chunk_converter_licensed@{conv.spark_output_column_names[0].split("@")[0]}')
            else:
                pipe.add(conv, name_to_add=f'chunk_converter@{conv.spark_output_column_names[0].split("@")[0]}')
        return pipe

    @staticmethod
    def enforce_AT_schema_on_embedding_processors(pipe):
        """For every embedding provider and consumer, enforce that their output col is named
        <output_level>@storage_ref for output_levels word,chunk,sentence aka document , i.e. word_embed@elmo or
        sentence_embed@elmo etc. """
        for c in pipe.components:
            # Leave pretrained component_list models untouched
            if c.loaded_from_pretrained_pipe: continue
            if ComponentUtils.is_embedding_provider(c):
                if '@' not in c.spark_output_column_names[0]:
                    new_embed_AT_ref = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(c, 'output')
                    c.spark_output_column_names = [new_embed_AT_ref]
                    c.model.setOutputCol(new_embed_AT_ref)
            if ComponentUtils.is_embedding_consumer(c):
                input_embed_col = ComponentUtils.extract_embed_col(c)
                if '@' not in input_embed_col:
                    # TODO set storage ref for traianble model?
                    new_embed_AT_ref = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(c, 'input')
                    c.spark_input_column_names.remove(input_embed_col)
                    c.spark_input_column_names.append(new_embed_AT_ref)
                    c.model.setInputCols(c.spark_input_column_names)

        return pipe

    @staticmethod
    def enforce_NLU_columns_to_NLP_columns(pipe):
        """for every component, set its inputs and outputs to the ones configured on the NLU component."""
        for c in pipe.components:
            if c.loaded_from_pretrained_pipe:
                continue
            if c.name == NLP_NODE_IDS.DOCUMENT_ASSEMBLER:
                continue
            c.model.setOutputCol(c.spark_output_column_names[0])
            if hasattr(c.model, 'setInputCols'):
                c.model.setInputCols(c.spark_input_column_names)
            else:
                # Some OCR Annotators only have one input and thus only setInputCol method but not setInputCols
                c.model.setInputCol(c.spark_input_column_names[0])
        return pipe

    @staticmethod
    def is_converter_component_resolution_reference(reference: str) -> bool:
        if 'chunk_emb' in reference:
            return True

    @staticmethod
    def configure_component_output_levels_to_sentence(pipe):
        '''
        Configure component_list components to output level document. Substitute every occurrence of <document> to
        <sentence> for every component that feeds from <document :param pipe: component_list to be configured
        :return: configured component_list
        '''
        logger.info('Configuring components to sentence level')
        for c in pipe.components:
            # update in/out spark cols
            if NLP_FEATURES.DOCUMENT in c.spark_input_column_names and NLP_FEATURES.SENTENCE not in c.spark_input_column_names and NLP_FEATURES.SENTENCE not in c.spark_output_column_names:
                logger.info(f"Configuring C={c.name}  of Type={type(c.model)} to Sentence Level")
                c.spark_input_column_names.remove(NLP_FEATURES.DOCUMENT)
                c.spark_input_column_names.append(NLP_FEATURES.SENTENCE)
                c.model.setInputCols(c.spark_input_column_names)
                c.output_level = NLP_LEVELS.SENTENCE
            # update in/out col types
            if NLP_FEATURES.DOCUMENT in c.in_types and NLP_FEATURES.SENTENCE not in c.in_types and NLP_FEATURES.SENTENCE not in c.out_types:
                c.in_types.remove(NLP_FEATURES.DOCUMENT)
                c.in_types.append(NLP_FEATURES.SENTENCE)
        return pipe.components

    @staticmethod
    def configure_component_output_levels_to_document(pipe):
        '''
        Configure component_list components to output level document. Substitute every occurence of <sentence> to <document> for every component that feeds from <sentence>
        :param pipe: component_list to be configured
        :return: configured component_list coonents only
        '''
        logger.info('Configuring components to document level')
        for c in pipe.components:
            # Update in/out spark cols
            if NLP_FEATURES.SENTENCE in c.spark_input_column_names and NLP_FEATURES.DOCUMENT not in c.spark_input_column_names and NLP_FEATURES.DOCUMENT not in c.spark_output_column_names:
                logger.info(f"Configuring C={c.name} to document output level")
                c.spark_input_column_names.remove(NLP_FEATURES.SENTENCE)
                c.spark_input_column_names.append(NLP_FEATURES.DOCUMENT)
                c.model.setInputCols(c.spark_input_column_names)
                c.output_level = NLP_LEVELS.DOCUMENT
            # Update in/out col types
            if NLP_FEATURES.SENTENCE in c.in_types and NLP_FEATURES.DOCUMENT not in c.in_types and NLP_FEATURES.DOCUMENT not in c.out_types:
                c.in_types.remove(NLP_FEATURES.SENTENCE)
                c.in_types.append(NLP_FEATURES.DOCUMENT)
        return pipe.components

    @staticmethod
    def has_sentence_detector(pipe):
        """Check for NLUPipieline if it contains sentence detector"""
        for c in pipe.components:
            if isinstance(c.model, (SentenceDetectorDLModel, SentenceDetector, SentenceDetectorDLApproach)): return True
        return False

    @staticmethod
    def configure_component_output_levels(pipe, force_level=''):
        '''
        This method configures sentenceEmbeddings and Classifier components to output at a specific level
        This method is called the first time .predit() is called and every time the output_level changed
        If output_level == Document, then sentence embeddings will be fed on Document col and classifiers recieve doc_embeds/doc_raw column, depending on if the classifier works with or withouth embeddings
        If output_level == sentence, then sentence embeddings will be fed on sentence col and classifiers recieve sentence_embeds/sentence_raw column, depending on if the classifier works with or withouth embeddings. IF sentence detector is missing, one will be added.
        component_list : NLU pipeline
        return : a new list of components, which columns cnfigured for output level
        '''
        if pipe.output_level == 'sentence' or force_level == 'sentence':
            if not PipeUtils.has_sentence_detector(pipe):
                logger.info("Adding missing Sentence Detector")
                pipe.is_fitted = False
                sentence_detector = ComponentMap.os_components[NLP_NODE_IDS.SENTENCE_DETECTOR_DL]
                sentence_detector.set_metadata(sentence_detector.get_default_model(), 'detect_sentence',
                                               'sentence_detector_dl', 'en', False, Licenses.open_source)
                pipe.components.insert(1, sentence_detector)
            return PipeUtils.configure_component_output_levels_to_sentence(pipe)
        elif pipe.output_level == 'document' or force_level == 'document':
            return PipeUtils.configure_component_output_levels_to_document(pipe)

    @staticmethod
    def check_if_component_is_in_pipe(pipe, component_name_to_check, check_strong=True):
        """Check if a component with a given name is already in a component_list """
        for c in pipe.components:
            if check_strong and component_name_to_check == c.info.name:
                return True
            elif not check_strong and component_name_to_check in c.info.name:
                return True
        return False

    @staticmethod
    def check_if_there_component_with_col_in_components(component_list, features, except_component):
        """For a given list of features and a list of components, see if there are components taht provide this feature
        If yes, True, otherwise False
        """
        for c in component_list:
            if c.out_types[0] != except_component.out_types[0]:
                for f in ComponentUtils.clean_irrelevant_features(c.info.spark_output_column_names, True):
                    if f in features: return True

        return False

    @staticmethod
    def is_leaf_node(c, pipe) -> bool:
        """Check if a component is a leaf in the DAG.
        We verify by checking if any other_c is feeding from os_components.
        If yes, it is not a leaf. If nobody feeds from os_components, it's a leaf.
        """
        inputs = c.info.inputs
        for other_c in pipe.components:
            if c is not other_c:
                for f in other_c.info.inputs: 1

        return False

    @staticmethod
    def clean_AT_storage_refs(pipe):
        """Removes AT notation from all columns. Useful to reset component_list back to default state"""
        for c in pipe.components:
            if c.info.loaded_from_pretrained_pipe:
                continue
            c.info.inputs = [f.split('@')[0] for f in c.info.inputs]
            c.out_types = [f.split('@')[0] for f in c.out_types]
            c.info.spark_input_column_names = [f.split('@')[0] for f in c.info.spark_input_column_names]
            c.info.spark_output_column_names = [f.split('@')[0] for f in c.info.spark_output_column_names]
            c.info.spark_input_column_names = c.info.inputs.copy()
            c.info.spark_output_column_names = c.out_types.copy()

        return pipe

    @staticmethod
    def rename_duplicate_cols(pipe):
        """Rename cols with duplicate names"""
        for i, c in enumerate(pipe.components):
            for other_c in pipe.components:
                if c is other_c:
                    continue
                if c.loaded_from_pretrained_pipe:
                    continue
                if c.spark_output_column_names[0] == other_c.spark_output_column_names[0]:
                    c.spark_output_column_names[0] = f'{c.spark_output_column_names[0]}_{str(i)}'
        return pipe

    @staticmethod
    def find_trainable_embed_consumer(pipe):
        """Find traianble component which consumes emeddings.
        Returns index of component and type of embedding if found, otherwise returns -1 and None"""
        for i, c in enumerate(pipe.components):
            if c.trainable and c.has_storage_ref:
                return pipe.components.index(c), ComponentUtils.extract_embed_col(c, 'input')

        return -1, None

    @staticmethod
    def remove_convertable_storage_refs(required_features_ref, conversion_candidates, provided_features_ref):
        """Remove required storage ref features if conversion candidate has it, so that storage ref provider will not
        be downloaded twice """
        if len(conversion_candidates) == 0:
            return required_features_ref, conversion_candidates
        # ComponentUtils.extract_storage_ref_AT_notation_for_embeds
        for candidate in conversion_candidates:
            # candidate_at_storage_ref_feature = ComponentUtils.extract_storage_ref_AT_notation_for_embeds(
            #     candidate.component_candidate, 'output')
            if candidate.component_candidate is None:
                continue
            for feature in required_features_ref:
                # if feature not in provided_features_ref: # TODO revisit this after deep test
                #     # Feature not yet manifested by creating corresponding anno
                #     # Unless its also a storage ref candidate. In this scenario, the Feature is manifested but the Converter is missing.
                #     # Remove the feature from requirements, since its already there and will otherwise cause storage ref resolution to manifest again
                #     continue
                required_storage_ref = feature.split('@')[-1]
                if required_storage_ref == candidate.storage_ref:  # or candidate_at_storage_ref_feature == feature
                    # The feature is already provided, but not converted. We can remove it
                    required_features_ref.remove(feature)

        return required_features_ref, conversion_candidates

    @staticmethod
    def update_converter_storage_refs_and_cols(pipe, provided_features_ref, required_features_ref):
        """Storage ref of converters is initially empty string, i.e. '' .
        This method checks if  any convertable embeddings are provided, if yes it will update storage ref of converter
        , update the input/output columns with colname@storage_ref notation and mark it as resolved
        by removing it from the corrosponding lists"""

        for c in pipe.components:
            if c.name in [NLP_NODE_IDS.SENTENCE_EMBEDDINGS_CONVERTER, NLP_NODE_IDS.CHUNK_EMBEDDINGS_CONVERTER]:
                # Check if there are candidates that feed the converter, any word Embedding will work
                if c.storage_ref != '':
                    # If storage_ref is not '' then this is converter is already fixed, nothing to do
                    continue
                for other_c in pipe.components:
                    if other_c.has_storage_ref and other_c.type == AnnoTypes.TOKEN_EMBEDDING:
                        # Get original embed cols
                        in_embed = ComponentUtils.extract_embed_col(c, 'input')
                        out_embed = ComponentUtils.extract_embed_col(c, 'output')

                        if len(in_embed.split('@')) == 2:
                            # Storage ref is already on annotator, we dont ned to fix this
                            continue

                        c.spark_output_column_names.remove(out_embed)
                        c.spark_input_column_names.remove(in_embed)
                        provided_features_ref.remove(out_embed + '@')
                        required_features_ref.remove(in_embed + '@')
                        storage_ref = StorageRefUtils.extract_storage_ref(other_c)
                        in_embed = in_embed + '@' + storage_ref
                        out_embed = out_embed + '@' + storage_ref
                        c.spark_output_column_names.append(out_embed)
                        c.spark_input_column_names.append(in_embed)
                        provided_features_ref.append(out_embed)
                        required_features_ref.append(in_embed)
                        c.storage_ref = storage_ref

        return provided_features_ref, required_features_ref

    @staticmethod
    def add_metadata_to_pipe(pipe):
        """Write metadata fields to pipeline, for now only whether it contains
        OCR components or not. To be extended in the future
        """
        py_class_to_anno_id = AnnoClassRef.get_ocr_pyclass_2_anno_id_dict()

        for c in pipe.components:
            # Check for OCR componments
            if c.jsl_anno_py_class in py_class_to_anno_id.keys():
                pipe.contains_ocr_components = True

        for c in pipe.components:
            # Check for licensed components
            if c.license in [Licenses.ocr, Licenses.hc]:
                pipe.has_licensed_components = True
        return pipe
