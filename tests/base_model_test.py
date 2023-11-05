from dataclasses import dataclass
from typing import Optional, List

import pytest

from tests.test_utils import model_and_output_levels_test


@dataclass
class NluTest:
    nlu_ref: str
    lang: str
    test_group: str
    input_data_type: str
    library: str
    output_levels: Optional[List[str]] = None


models_to_test = [
    NluTest(nlu_ref="chunk", lang='en', test_group='chunker', input_data_type='generic', library='open_source'),
    NluTest(nlu_ref="ngram", lang='en', test_group='chunker', input_data_type='generic', library='open_source'),
    NluTest(nlu_ref="zh.segment_words", lang='zh', test_group='tokenizer', input_data_type='generic',
            library='open_source'),
    NluTest(nlu_ref="zh.tokenize", lang='zh', test_group='tokenizer', input_data_type='generic',
            library='open_source'),
    NluTest(nlu_ref="tokenize", lang='en', test_group='tokenizer', input_data_type='generic',
            library='open_source'),
    NluTest(nlu_ref="en.assert.biobert", lang='en', test_group='assertion', input_data_type='medical',
            library='healthcare'),
    NluTest(nlu_ref="relation.drug_drug_interaction", lang='en', test_group='relation', input_data_type='medical',
            output_levels=['chunk', 'tokens', 'embeddings', 'document', 'relation'], library='healthcare'),
    NluTest(nlu_ref="pdf2table", lang='en', test_group='table_extractor', input_data_type='PPT_table',
            library='ocr'),
    NluTest(nlu_ref="ppt2table", lang='en', test_group='table_extractor', input_data_type='PDF_table',
            library='ocr'),
    NluTest(nlu_ref="doc2table", lang='en', test_group='table_extractor', input_data_type='DOC_table',
            library='ocr'),
]


@pytest.mark.parametrize("model_to_test", models_to_test)
def test_model(model_to_test: NluTest):
    model_and_output_levels_test(
        nlu_ref=model_to_test.nlu_ref,
        lang=model_to_test.lang,
        test_group=model_to_test.test_group,
        output_levels=model_to_test.output_levels,
        input_data_type=model_to_test.input_data_type,
        library=model_to_test.library
    )
