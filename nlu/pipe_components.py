

from nlu import  *
class NLUComponent():
    '''
        This class loads all the components in the components folder.
        It acts as an accessor to every component.
        Pass a string identifier to it and it will construct and return the model.
        It will also take care of setting the initial configuration parameter dict .
     '''
    def __init__(self, component_name, component_type):

        self.model = None # Either Spark NLP model or some 3rd party custom model. Reference to a model
        self.component_path = nlu.nlu_package_location + 'components/' + component_type + 's/' + component_name + '/'
        self.component_info = nlu.ComponentInfo.from_directory(component_info_dir= self.component_path)


    def set_parameter_on_model(self,key, value  ): pass # Implemented by child class which is extending from NLU_component

    def print_parameters_explanation(self): pass
    def print_parameters(self): pass
    def info(self): pass




class SparkNLUComponent(NLUComponent):
    def __init__(self, component_name, component_type):
        # super().__init__(component_name, component_type)
        # super(SparkNLUComponent,self).__init__(component_name, component_type)
        NLUComponent.__init__(self, component_name, component_type)
        self.spark = nlu.sparknlp.start()
        self.nlu_reference=''
        nlu.spark = self.spark
        nlu.spark_started = True

class Component():
    # returns a component for a given name
    @staticmethod
    def __call__( name):
        component_info = nlu.AllComponentsInfo.get_component_info_by_name(name)

        if component_info.type == 'tokenizer' : return Tokenizer(name)
        if component_info.type == 'embedding' : return Embeddings(name)
        if component_info.type == 'classifier' : return Classifier(name)
        if component_info.type == 'labeled_dependency_parser' : return LabledDepParser(name)
        if component_info.type == 'unlabeled_dependency_parser' : return UnlabledDepParser(name)
        if component_info.type == 'lemmatizer' : return Lemmatizer(name)
        if component_info.type == 'normalizer' : return Normalizer(name)
        if component_info.type == 'spell_checker' : return SpellChecker(name)
        if component_info.type == 'util' : return Util(name)
