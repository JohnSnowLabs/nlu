BASE_CODES = {
'SIMILARITY':"""
nlu.load({}).viz_streamlit_word_similarity(['{}', '{}'])""",

'TOKEN':"""
nlu.load({}).viz_streamlit_token({})""",

'TREE':"""
nlu.load({}).viz_streamlit_dep_tree({})""",

'NER':"""
nlu.load({}).viz_streamlit_ner({})""",

'CLASSES':"""
nlu.load({}).viz_streamlit_classes({})""",

'FULL_UI':"""
????
""",


'PREFIX': """
import nlu
""",

'INSTALL':"""
[Refer to NLU install docs for installing NLU](https://nlu.johnsnowlabs.com/docs/en/install)
""",

}
from  typing import List,Union
def get_code_for_viz(viz_type:str,models:Union[str, List[str]],data:Union[str,List[str]],max_len:int=100)->str:
    """Generate code sample for displaying how to generate visualization"""
    if viz_type =='SIMILARITY':
        if len(data) >= 2 and isinstance(data[0],str) and isinstance(data[0],str):
            models = f"'{' '.join(models)}'"
            return BASE_CODES[viz_type].format(models, data[0][:max_len], data[1][:max_len])
    else :
        models = f"'{models}'" if isinstance(models, str)  else f"'{' '.join(models)}'"
        data = f"'{data[:max_len]}'"
        return  BASE_CODES[viz_type].format(models, data)

