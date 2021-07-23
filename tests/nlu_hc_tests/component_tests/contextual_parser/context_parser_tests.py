import unittest
import tests.nlu_hc_tests.secrets as sct
import nlu
import nlu.pipe.pipe_components
from sparknlp.annotator import *
from sparknlp_jsl.annotator import ContextualParserApproach,ContextualParserModel
from typing import *

from dataclasses import dataclass, field

@dataclass
class EntityDefinition:
    """
    case class EntityDefinition(
                            entity: String,
                            ruleScope: String,
                            regex: Option[String],
                            contextLength: Option[Double],
                            prefix: Option[List[String]],
                            var regexPrefix: Option[String],
                            suffix: Option[List[String]],
                            var regexSuffix: Option[String],
                            context: Option[List[String]],
                            contextException: Option[List[String]],
                            exceptionDistance: Option[Double],
                            var regexContextException: Option[String],
                            matchScope: Option[String],
                            completeMatchRegex: Option[String]


)
    val annotation = Annotation(outputAnnotatorType, matchedToken.begin, matchedToken.end, matchedToken.valueMatch,
      Map("field" -> $$(entityDefinition).entity, ### ENTITY CLASS
        "normalized" -> normalizedValue,
        "confidenceValue" -> BigDecimal(matchedToken.confidenceValue).setScale(2, BigDecimal.RoundingMode.HALF_UP).toString,
        "hits" -> matchedToken.hits,
        "sentence" -> matchedToken.sentenceIndex.toString))

    """
    # TODO Dictopmary {ara,eter define the set of words that you want to match and the word that will replace this match.
    entity: str   # The name of this rule
    regex: Optional[str] # Regex Pattern to extract candidates
    contextLength : Optional[int] # defines the maximum distance a prefix and suffix words can be away from the word to match,whereas context are words that must be immediately after or before the word to match
    prefix : Optional[List[str]] # Words preceding the regex match, that are at most `contextLength` characters aways
    regexPrefix : Optional[str]  # RegexPattern of words preceding the regex match, that are at most `contextLength` characters aways
    suffix : Optional[List[str]]  # Words following the regex match, that are at most `contextLength` characters aways
    regexSuffix : Optional[str] # RegexPattern of words following the regex match, that are at most `contextLength` distance aways
    context : Optional[List[str]] # list of words that must be immediatly before/after a match
    contextException : Optional[List[str]] #  ?? List of words that may not be immediatly before/after a match
    exceptionDistance : Optional[int] # Distance exceptions must be away from a match
    regexContextException : Optional[str] # Regex Pattern of exceptions that may not be within `exceptionDistance` range of the match
    matchScope : Optional[str] # Either `token` or `sub-token` to match on character basis
    completeMatchRegex : Optional[str] # Wether to use complete or partial matching, either `"true"` or `"false"`
    ruleScope: str # currently only `sentence` supported


class ContextParserTests(unittest.TestCase):

    def dump_dict_to_json_file(self, dict, path):
        """Generate json with dict contexts at target path"""
        import json
        with open(path, 'w') as f: json.dump(dict, f)

    def test_dict_dump(self):
        p = '/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tmp/trasgh/data.json'
        d = {'entity': "Temperature",
             'ruleScope': "sentence",
             'matchScope': "token",
             'regex': "\\b((9[0-9])|(10[0-9]))((\\.|,)[0-9]+)?\\b",
             'prefix': ["temperature", "fever"], #
             'suffix': ["Fahrenheit", "Celsius", "centigrade", "F", "C"],
             'contextLength': 30}
        ContextParserTests().dump_dict_to_json_file(d, p)


    def dump_data(self):
        gender = '''male,man,male,boy,gentleman,he,him
        female,woman,female,girl,lady,old-lady,she,her
        neutral,neutral'''

        with open('/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tmp/trasgh/gender.csv', 'w') as f:
            f.write(gender)


        gender = {
            "entity": "Gender",
            "ruleScope": "sentence",
            "completeMatchRegex": "true"
        }

        import json

        with open('/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tmp/trasgh/gender.json', 'w') as f:
            json.dump(gender, f)


    def test_context_parser(self):

        """

        - contextLength  defines the maximum distance a prefix and suffix words can be away from the word to match,whereas context are words that must be immediately after or before the word to match

         - dictionary parameter. In this parameter, you define the set of words that you want to match and the word that will replace this match.
        :return:
        """



        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET)
        ContextParserTests().dump_data()
        data = 'Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain'
        contex_pipe = nlu.load('match.context')
        contex_pipe.print_info()
        contex_pipe['context_matcher'].setCaseSensitive(False)         #| Info: Whether to use case sensitive when matching values | Currently set to : False
        contex_pipe['context_matcher'].setPrefixAndSuffixMatch(False)  #| Info: Whether to match both prefix and suffix to annotate the hit | Currently set to : False
        contex_pipe['context_matcher'].setContextMatch(False)           #| Info: Whether to include context to annotate the hit | Currently set to : True
        # contex_pipe['context_matcher'].setUpdateTokenizer(True)        #| Info: Whether to update tokenizer from pipeline when detecting multiple words on dictionary values | Currently set to : True
        contex_pipe['context_matcher'].setJsonPath('/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tmp/trasgh/gender.json')
        contex_pipe['context_matcher'].setDictionary('/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit2/tmp/trasgh/gender.csv', read_as=ReadAs.TEXT, options={"delimiter":","})

        # contex_pipe['parse_context'].SET_SMTH



        data = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to 
    presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis 
    three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index 
    ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting.
    Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . 
    She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . 
    She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was 
    significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , 
    or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , 
    anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin 
    ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed 
    as blood samples kept hemolyzing due to significant lipemia .
    The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior 
    to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , 
    the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , 
    and lipase was 52 U/L .
     β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged 
     and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . 
     The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides 
     to 1400 mg/dL , within 24 hours .
     Twenty days ago.
     Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . 
     At birth the typical boy is growing slightly faster than the typical girl, but the velocities become equal at about 
     seven months, and then the girl grows faster until four years. 
     From then until adolescence no differences in velocity 
     can be detected. 21-02-2020 
    21/04/2020
    """


        res = contex_pipe.predict(data, metadata=True, )  # .predict(data)

        print(res.columns)
        for c in res: print(res[c])
        print(res)


if __name__ == '__main__':
    ContextParserTests().test_entities_config()
