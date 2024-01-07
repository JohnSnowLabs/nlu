import pytest

from tests.utils import all_tests, one_per_lib, NluTest, model_and_output_levels_test


def model_id(model_to_test: NluTest) -> str:
    return f"{model_to_test.test_group}_{model_to_test.nlu_ref}"


def all_annotator_tests():
    return all_tests


def one_test_per_lib():
    return one_per_lib


@pytest.mark.skip(reason="Use run_tests.py instead until pytest-xdist issue is fixed")
@pytest.mark.parametrize("model_to_test", all_annotator_tests(), ids=model_id)
def test_model_all_annotators(model_to_test: NluTest):
    model_and_output_levels_test(
        nlu_ref=model_to_test.nlu_ref,
        lang=model_to_test.lang,
        test_group=model_to_test.test_group,
        output_levels=model_to_test.output_levels,
        input_data_type=model_to_test.input_data_type,
        library=model_to_test.library,
        pipe_params=model_to_test.pipe_params
    )


@pytest.mark.skip(reason="Local testing")
@pytest.mark.parametrize("model_to_test", one_test_per_lib(), ids=model_id)
def test_one_per_lib(model_to_test: NluTest):
    model_and_output_levels_test(
        nlu_ref=model_to_test.nlu_ref,
        lang=model_to_test.lang,
        test_group=model_to_test.test_group,
        output_levels=model_to_test.output_levels,
        input_data_type=model_to_test.input_data_type,
        library=model_to_test.library,
        pipe_params=model_to_test.pipe_params
    )
