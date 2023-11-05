import pytest

from tests.test_utils import model_and_output_levels_test

models_to_test = [
    ("chunk", 'en', 'chunker'
     , 'generic', None, 'open_source'),
    ("ngram", 'en', 'chunker', 'generic', None, 'open_source'),
    ("zh.segment_words", 'zh', 'tokenizer', 'generic', None, 'open_source'),
    ("zh.tokenize", 'zh', 'tokenizer', 'generic', None, 'open_source'),
    ("tokenize", 'en', 'tokenizer', 'generic', None, 'open_source'),
    ("en.assert.biobert", 'en', 'assertion', 'medical', None, 'healthcare'),
    ("en.assert.biobert", 'en', 'assertion', 'medical', None, 'healthcare'),
    ("relation.drug_drug_interaction", 'en', 'relation', 'medical',
     ['chunk', 'tokens', 'embeddings', 'document', 'relation'],
     'healthcare'),
    ("pdf2table", 'en', 'table_extractor', 'PPT_table', None, 'ocr'),
    ("ppt2table", 'en', 'table_extractor', 'PDF_table', None, 'ocr'),
    ("doc2table", 'en', 'table_extractor', 'DOC_table', None, 'ocr'),

]


@pytest.mark.parametrize("nlu_ref, lang, test_group, input_data_type, output_levels, library", models_to_test)
def test_model(nlu_ref, lang, test_group, input_data_type, output_levels, library):
    model_and_output_levels_test(nlu_ref=nlu_ref, lang=lang, test_group=test_group, output_levels=output_levels,
                                 input_data_type=input_data_type, library=library)
