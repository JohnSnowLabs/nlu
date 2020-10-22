from nlu.pipe_components import SparkNLUComponent, NLUComponent


class UnlabeledDependencyParser(SparkNLUComponent):

    def __init__(self, annotator_class='unlabeled_dependency_parser', language='en', component_type='dependency_untyped', get_default = True, nlp_ref='', nlu_ref ='', model=None):

        if model != None :self.model = model
        elif 'dep' in annotator_class or 'dep.untyped' in annotator_class or annotator_class== 'unlabeled_dependency_parser':
            from nlu.components.dependency_untypeds.unlabeled_dependency_parser.unlabeled_dependency_parser import UnlabeledDependencyParser
            if get_default : self.model = UnlabeledDependencyParser.get_default_model()
            else : self.model = UnlabeledDependencyParser.get_pretrained_model(nlp_ref, language)


        SparkNLUComponent.__init__(self, annotator_class, component_type)
