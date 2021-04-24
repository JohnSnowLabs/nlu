from sparknlp.annotator import *
from nlu.pipe.viz.vis_utils_OS import VizUtilsOS

class VizUtils():
    """Utils for interfacing with the Spark-NLP-Display lib"""
    @staticmethod
    def infer_viz_type(pipe)->str:
        """For a given NLUPipeline, infers which visualizations are applicable. """
        if pipe.has_licensed_components :
            from nlu.pipe.viz.vis_utils_HC import VizUtilsHC
            return VizUtilsHC.infer_viz_licensed(pipe)
        else : return VizUtilsOS.infer_viz_open_source(pipe)


    @staticmethod
    def viz_OS(anno_res, pipe, viz_type,viz_colors,labels_to_viz,is_databricks_env):
        """Vizualize open source component"""
        if   viz_type == 'ner' : VizUtilsOS.viz_ner(anno_res, pipe,labels_to_viz,viz_colors,is_databricks_env)
        elif viz_type == 'dep' : VizUtilsOS.viz_dep(anno_res, pipe,is_databricks_env)
        else : raise ValueError("Could not find applicable viz_type. Please make sure you specify either ner, dep, resolution, relation, assert or dep and have loaded corrosponding components")

    @staticmethod
    def viz_HC(anno_res, pipe, viz_type,viz_colors,labels_to_viz,is_databricks_env):
        """Vizualize licensed component"""
        from nlu.pipe.viz.vis_utils_HC import VizUtilsHC
        if   viz_type == 'ner' : VizUtilsHC.viz_ner(anno_res, pipe,labels_to_viz,viz_colors,is_databricks_env)
        elif viz_type == 'dep' : VizUtilsHC.viz_dep(anno_res, pipe,is_databricks_env)
        elif viz_type == 'resolution' : VizUtilsHC.viz_resolution(anno_res, pipe,viz_colors,is_databricks_env)
        elif viz_type == 'relation' : VizUtilsHC.viz_relation(anno_res, pipe,is_databricks_env)
        elif viz_type == 'assert' : VizUtilsHC.viz_assertion(anno_res, pipe,viz_colors,is_databricks_env)
        else : raise ValueError("Could not find applicable viz_type. Please make sure you specify either ner, dep, resolution, relation, assert or dep and have loaded corrosponding components")

"""Define whiche annotators model are definable by which vizualizer. There are 5 in total, 2 open source and 5 HC"""
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