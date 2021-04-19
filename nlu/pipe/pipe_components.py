# from nlu import *
import nlu
class NLUComponent():
    '''
        This class loads all the components in the components folder.
        It acts as an accessor to every component.
        Pass a string identifier to it and it will construct and return the model.
        It will also take care of setting the initial configuration parameter dict .
     '''

    def __init__(self, component_name, component_type):
        # because we ae using the input/output/label fix, we run supers init AFTER model was created, so the next statement would overwrite it
        # when NLU might support 3rd party models we should rework this
        # self.model = None  # Either Spark NLP model or some 3rd party custom model. Reference to a model
        self.component_path = nlu.nlu_package_location + 'components/' + component_type + 's/' + component_name + '/'
        self.info = nlu.ComponentInfo.from_directory(component_info_dir=self.component_path)

    def print_parameters_explanation(self):
        pass

    def print_parameters(self):
        pass

    def info(self):
        print(self.info['info'])
        self.print_parameters_explanation()
        self.print_parameters()


class  SparkNLUComponent(NLUComponent):
    def __init__(self, component_name, component_type, nlu_ref='', nlp_ref='',lang='',loaded_from_pretrained_pipe=False, is_licensed=False):
        NLUComponent.__init__(self, component_name, component_type)
        self.info.nlu_ref = nlu_ref
        self.info.nlp_ref = nlp_ref
        self.info.lang    = lang
        self.info.loaded_from_pretrained_pipe = loaded_from_pretrained_pipe
        self.__set_missing_model_attributes__()
        if is_licensed : self.info.license = 'healthcare'

    def __set_missing_model_attributes__(self):
        '''
        For a given Spark NLP model this model will extract the poarameter map and search for input/output/label columns and set them on the model.
        This is a workaround to undefined behaviour when getting input/output/label columns
        :param : The model for which the attributes should be set
        :return: model with attributes properly set
        '''
        for k in self.model.extractParamMap():
            if "inputCol" in str(k):
                if isinstance(self.model.extractParamMap()[k], str) :
                    if self.model.extractParamMap()[k] == 'embeddings': # swap name so we have uniform col names
                        self.model.setInputCols( 'word_embeddings')
                    self.info.spark_input_column_names =  [self.model.extractParamMap()[k]]
                else :
                    if 'embeddings' in self.model.extractParamMap()[k]: # swap name so we have uniform col names
                        new_cols = self.model.extractParamMap()[k]
                        new_cols.remove("embeddings")
                        new_cols.append("word_embeddings")
                        self.model.setInputCols(new_cols)
                    self.info.spark_input_column_names =  self.model.extractParamMap()[k]

            if "outputCol" in str(k):
                if isinstance(self.model.extractParamMap()[k], str) :
                    if self.model.extractParamMap()[k] == 'embeddings': # swap name so we have uniform col names
                        self.model.setOutputCol( 'word_embeddings')
                    self.info.spark_output_column_names =  [self.model.extractParamMap()[k]]
                else :
                    if 'embeddings' in self.model.extractParamMap()[k]: # swap name so we have uniform col names
                        new_cols = self.model.extractParamMap()[k]
                        new_cols.remove("embeddings")
                        new_cols.append("word_embeddings")
                        self.model.setOutputCol(new_cols)
                    self.info.spark_output_column_names =  self.model.extractParamMap()[k]





            # if "labelCol" in str(k):
            #     if isinstance(self.model.extractParamMap()[k], str) :
            #         self.component_info['spark_label_column_names'] =  [self.model.extractParamMap()[k]]
            #     else :
            #         self.component_info['spark_label_column_names'] =  self.model.extractParamMap()[k]
