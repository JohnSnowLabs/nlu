from typing import Dict, Any, List
import logging
from nlu.universe.atoms import NlpLevel
from nlu.universe.feature_node_ids import NLP_HC_NODE_IDS
from nlu.universe.logic_universes import NLP_LEVELS
from nlu.universe.feature_universes import NLP_FEATURES
from nlu.universe.universes import Licenses
from nlu.pipe.col_substitution.col_name_substitution_utils import ColSubstitutionUtils
logger = logging.getLogger('nlu')


class OutputLevelUtils:
    """Resolve output level of pipeline and components"""

    @staticmethod
    def infer_prediction_output_level(pipe) -> NlpLevel:
        """
        This function checks the LAST  component_to_resolve of the NLU pipeline and infers
         from that the output level via checking the components' info.
        :param pipe: to infer output level for
        :return returns inferred output level
        """
        # Loop in reverse over component_list and get first output level
        # (???) of non util/sentence_detector/tokenizer/doc_assembler.
        for c in pipe.components[::-1]:
            return OutputLevelUtils.resolve_component_to_output_level(pipe, c)
        # fallback
        return NLP_LEVELS.DOCUMENT

    @staticmethod
    def resolve_input_dependent_component_to_output_level(pipe, component_to_resolve) -> NlpLevel:
        """
        For a given NLU component  which is input dependent , resolve its output level by checking if it's input stem
        from document or sentence based annotators
        :param pipe: the pipeline containing all components
        :param component_to_resolve:  Input dependent component for which we want to know the output level
        :return: output-level of component
        """
        # (1.) A classifier, which is using sentence/document. We just check input cols
        if NLP_FEATURES.DOCUMENT in component_to_resolve.spark_input_column_names:
            return NLP_LEVELS.DOCUMENT
        if NLP_FEATURES.SENTENCE in component_to_resolve.spark_input_column_names:
            return NLP_LEVELS.SENTENCE

        # (2.) A model_anno_obj which is input dependent and not using document/sentence cols
        # We iterator over components and see which is feeding this input dependent component_to_resolve
        for c in pipe.components:
            if c.name == component_to_resolve.name:
                continue
            if c.spark_output_column_names[0] in component_to_resolve.spark_input_column_names:
                # We found a component that is feeding the component_to_resolve.
                # Now we need to check if that component is document/sentence level
                if NLP_LEVELS.DOCUMENT in c.spark_input_column_names:
                    return NLP_LEVELS.DOCUMENT
                elif NLP_LEVELS.SENTENCE in c.spark_input_column_names:
                    return NLP_LEVELS.SENTENCE

    @staticmethod
    def resolve_component_to_output_level(pipe, component) -> NlpLevel:
        """
        For a given NLU component_to_resolve, resolve its output level,
        by checking annotator_levels dicts for approaches and models
        If output level is input dependent, resolve_input_dependent_component_to_output_level will resolve it
        :param component:  to resolve
        :param pipe:  pipe containing the component
        :return: resolve component_to_resolve
        """
        if 'input_dependent' in component.output_level:
            return OutputLevelUtils.resolve_input_dependent_component_to_output_level(pipe, component)
        else:
            return component.output_level

    @staticmethod
    def get_columns_at_same_level_of_pipe(pipe, df, anno_2_ex_config, get_embeddings) -> List[str]:
        """Get List of columns in df that are generated from components in the pipeline
            which are at the same output level as the pipe .
        :param pipe: NLU Pipeline
        :param df: Pandas DataFrame resulting from applying the pipe
        :param anno_2_ex_config: mapping between anno to extractor, from get_annotator_extraction_configs()
        :param get_embeddings: Should embeddings be included
        :return: List of columns which are generated from components
                at same output level as the pipe.prediction_output_level
        """
        same_output_level_cols = []
        for c in pipe.components:
            if 'embedding' in c.type and get_embeddings is False:
                continue
            output_level = OutputLevelUtils.resolve_component_to_output_level(pipe, c)
            if output_level == pipe.prediction_output_level:
                generated_cols = ColSubstitutionUtils.get_final_output_cols_of_component(c, df, anno_2_ex_config)
                for generated_col in generated_cols:
                    # if '_k_' in generated_col and c.jsl_anno_class_id == NLP_HC_NODE_IDS.SENTENCE_ENTITY_RESOLVER:
                    #     # all _k_ fields of resolver may never be viewed as any common outputlevel and thus never be zipped.
                    #     continue
                    same_output_level_cols.append(generated_col)
        return list(set(same_output_level_cols))

    @staticmethod
    def get_output_level_mapping_by_component(pipe) -> Dict[Any, str]:
        """Get a mapping key=NluComponent and value = output level
        :param pipe: NLU pipe for which to get the mapping
        :return: dict where key = NLU_Component and Value = Output level
        """
        nlp_levels = {c: OutputLevelUtils.resolve_component_to_output_level(pipe, c) for c in pipe.components}
        for c in pipe.components:
            if c.license == Licenses.ocr:
                nlp_levels[c] = c.output_level
        return {c: OutputLevelUtils.resolve_component_to_output_level(pipe, c) for c in pipe.components}
