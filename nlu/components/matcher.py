import nlu
from nlu.pipe.pipe_components import SparkNLUComponent


class Matcher(SparkNLUComponent):
    def __init__(self, annotator_class='date_matcher', language='en', component_type='matcher', get_default=False,
                 nlp_ref='', model=None, nlu_ref='', dataset='', is_licensed=False, loaded_from_pretrained_pipe=False):

        if 'date' in nlp_ref or 'date' in nlu_ref:
            annotator_class = 'date_matcher'
        elif 'regex' in nlp_ref or 'regex' in nlu_ref:
            annotator_class = 'regex_matcher'
        elif 'context' in nlu_ref:
            annotator_class = 'context_parser'
        elif 'text' in nlp_ref or 'text' in nlu_ref:
            annotator_class = 'text_matcher'
        elif '_matcher' not in annotator_class:
            annotator_class = annotator_class + '_matcher'
        if model != None:
            self.model = model
        else:
            if 'context' in annotator_class:
                from nlu.components.matchers.context_parser.context_parser import ContextParser
                is_licensed = True
                if get_default:
                    self.model = ContextParser.get_default_model()
                else:
                    self.model = ContextParser.get_default_model()

            elif 'text' in annotator_class:
                from nlu import TextMatcher
                if get_default or nlp_ref =='text_matcher':
                    self.model = TextMatcher.get_default_model()
                else:
                    self.model = TextMatcher.get_pretrained_model(nlp_ref, language)
            elif 'date' in annotator_class:
                from nlu import DateMatcher
                from nlu.components.matchers.date_matcher.date_matcher import DateMatcher as DateM

                if get_default: self.model = DateM.get_default_model()
                else: self.model = DateM.get_default_model()
            elif 'regex' in annotator_class:
                from nlu import RegexMatcher
                if get_default:
                    self.model = RegexMatcher.get_default_model()
                else:
                    self.model = RegexMatcher.get_pretrained_model(nlu_ref, language)

        SparkNLUComponent.__init__(self, annotator_class, component_type, nlu_ref, nlp_ref, language,loaded_from_pretrained_pipe , is_licensed)
