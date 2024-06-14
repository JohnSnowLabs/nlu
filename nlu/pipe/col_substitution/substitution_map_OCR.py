"""
Resolve Annotator Classes in the Pipeline to Extractor Configs and Methods
Every Annotator should have 2 configs. Some might offor multuple configs/method pairs, based on model_anno_obj/NLP reference.
- default/minimalistic -> Just the results of the annotations, no confidences or extra metadata
- with meta            -> A config that leverages white/black list and gets the most relevant metadata
- with positions       -> With Begins/Ends
- with sentence references -> Reeturn the sentence/chunk no. reference from the metadata.
                                If a document has multi-sentences, this will map a label back to a corrosponding sentence
"""
# from nlu.pipe.col_substitution.col_substitution_HC import *
from nlu.pipe.col_substitution.col_substitution_OS import *
from nlu.pipe.col_substitution.col_substitution_OCR import *

from sparkocr.transformers import *

OCR_anno2substitution_fn = {
    VisualDocumentClassifier : {
        'default': substitute_document_classifier_text_cols ,
    },
    VisualDocumentNerLilt : {
        'default': substitute_document_ner_cols,
    },
    FormRelationExtractor : {
        'default': substitute_form_extractor_text_cols,
    }

}
