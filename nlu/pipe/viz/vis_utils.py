from sparknlp.annotator import *
from nlu.pipe.viz.vis_utils_OS import VizUtilsOS
import random


class VizUtils():
    """Utils for interfacing with the Spark-NLP-Display lib"""

    @staticmethod
    def get_random():
        return random.randint(0, 1333333333337)

    @staticmethod
    def infer_viz_type(pipe) -> str:
        """For a given NLUPipeline, infers which visualizations are applicable. """
        if pipe.has_licensed_components:
            from nlu.pipe.viz.vis_utils_HC import VizUtilsHC
            return VizUtilsHC.infer_viz_licensed(pipe)
        else:
            return VizUtilsOS.infer_viz_open_source(pipe)

    @staticmethod
    def viz_OS(anno_res, pipe, viz_type, viz_colors, labels_to_viz, is_databricks_env, write_to_streamlit,
               streamlit_key,
               ner_col, pos_col, dep_untyped_col, dep_typed_col):
        """Vizualize open source component_to_resolve"""
        streamlit_key = VizUtils.get_random() if streamlit_key == "RANDOM" else streamlit_key
        if viz_type == 'ner':
            return VizUtilsOS.viz_ner(anno_res, pipe, labels_to_viz, viz_colors, is_databricks_env, write_to_streamlit,
                                      streamlit_key, ner_col)
        elif viz_type == 'dep':
            return VizUtilsOS.viz_dep(anno_res, pipe, is_databricks_env, write_to_streamlit, streamlit_key,
                                      pos_col, dep_untyped_col, dep_typed_col)
        else:
            raise ValueError(
                "Could not find applicable viz_type. Please make sure you specify either ner, dep, resolution, relation, assert or dep and have loaded corrosponding components")

    @staticmethod
    def viz_HC(anno_res, pipe, viz_type, viz_colors, labels_to_viz, is_databricks_env, write_to_streamlit,
               ner_col, pos_col, dep_untyped_col, dep_typed_col, resolution_col, relation_col, assertion_col):
        """Vizualize licensed component_to_resolve"""
        from nlu.pipe.viz.vis_utils_HC import VizUtilsHC
        if viz_type == 'ner':
            return VizUtilsHC.viz_ner(anno_res, pipe, labels_to_viz, viz_colors, is_databricks_env, write_to_streamlit,
                                      ner_col)
        elif viz_type == 'dep':
            return VizUtilsHC.viz_dep(anno_res, pipe, is_databricks_env, write_to_streamlit, dep_untyped_col,
                                      dep_typed_col, pos_col)
        elif viz_type == 'resolution':
            return VizUtilsHC.viz_resolution(anno_res, pipe, viz_colors, is_databricks_env,
                                             write_to_streamlit, ner_col, resolution_col)
        elif viz_type == 'relation':
            return VizUtilsHC.viz_relation(anno_res, pipe, is_databricks_env, write_to_streamlit, relation_col, )
        elif viz_type == 'assert':
            return VizUtilsHC.viz_assertion(anno_res, pipe, viz_colors, is_databricks_env, write_to_streamlit, ner_col,
                                            assertion_col)
        else:
            raise ValueError(
                "Could not find applicable viz_type. Please make sure you specify either ner, dep, resolution, relation, assert or dep and have loaded corrosponding components")


"""Define whiche annotators model_anno_obj are definable by which vizualizer. There are 5 in total, 2 open source and 5 HC"""
# vizalbe_components_OC = {
#     'ner' : [NerConverter],
#     'dep' : [DependencyParserModel],
# }

# vizalbe_components_HC = {
#     'ner':[NerConverter,NerConverterInternal],
#     'resolution' : [SentenceEntityResolverModel, ChunkEntityResolverModel] ,
#     'relation'   : [RelationExtractionModel,RelationExtractionDLModel],
#     'assert'     : [AssertionDLModel,AssertionLogRegApproach],
#     'dep' : [DependencyParserModel],
# }
