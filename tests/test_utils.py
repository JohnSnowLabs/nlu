import os
import sys

import pandas as pd
import sparknlp

import nlu
from tests.test_data import get_test_data

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


def log_df(df, test_group):
    if df is None:
        raise Exception('Cannot log Df which is none')
    if test_group == 'table_extractor':
        assert len(test_group) > 0, 'At least one table should have been extracted'
        for extracted_df in df:
            log_df(extracted_df, 'generic')
        return
    for c in df.columns:
        print(df[c])


def log_and_validate(df,test_group):
    log_df(df,test_group)
    validate_predictions(df)


def get_sample_pdf():
    data = {
        "text": [
            "This day sucks but tomorrow will be better ! ",
            "I love this day",
            "I dont like Sami",
        ]
    }
    text_df = pd.DataFrame(data)
    return text_df


def get_sample_pdf_with_labels():
    data = {
        "text": ["This day sucks", "I love this day", "I dont like Sami"],
        "sentiment_label": [1, 1, 0],
    }
    text_df = pd.DataFrame(data)
    return text_df


def get_sample_sdf():
    nlu.spark = sparknlp.start()
    nlu.spark_started = True
    return nlu.spark.createDataFrame(get_sample_pdf())


def get_sample_pdf_with_extra_cols():
    data = {
        "text": ["This day sucks", "I love this day", "I dont like Sami"],
        "random_feature1": [1, 1, 0],
        "random_feature2": ["d", "a", "3"],
    }
    text_df = pd.DataFrame(data)
    return text_df


def get_sample_pdf_with_no_text_col():
    data = {
        "schmext": ["This day sucks", "I love this day", "I dont like Sami"],
        "random_feature1": [1, 1, 0],
        "random_feature2": ["d", "a", "3"],
    }
    text_df = pd.DataFrame(data)
    return text_df


def get_sample_spark_dataframe():
    data = {
        "text": ["This day sucks", "I love this day", "I dont like Sami"],
        "random_feature1": [1, 1, 0],
        "random_feature2": ["d", "a", "3"],
    }
    text_df = pd.DataFrame(data)
    return text_df


def get_sample_pdf_with_extra_cols_and_entities():
    data = {
        "text": [
            "Pater says this day sucks. He lives in America. He likes Angela Merkel from Germany",
            "I love burgers from Burger King",
            "I dont like Sami, he lives in Asia",
        ],
        "random_feature1": [1, 1, 0],
        "random_feature2": ["d", "a", "3"],
    }
    text_df = pd.DataFrame(data)
    return text_df


import os
from os.path import expanduser


def download_dataset(
        data_url,
        output_file_name,
        output_folder,
):
    import urllib.request

    download_path = (
            create_dataset_dir_if_not_exist_and_get_path()
            + output_folder
            + output_file_name
    )

    # Check if dir exists, if not create it
    # create_path_if_not_exist(data_dir )
    create_path_if_not_exist(
        create_dataset_dir_if_not_exist_and_get_path() + output_folder
    )

    from pathlib import Path

    # Check if file exists, if not download it
    if not Path(download_path).is_file():
        urllib.request.urlretrieve(data_url, download_path)

    print("Downloaded dataset to ", download_path)
    return download_path


def create_dataset_dir_if_not_exist_and_get_path():
    root = expanduser("~")
    dataset_path = root + "/nlu_test_datasets/"
    if not os.path.exists(dataset_path):
        print("Creating dir", dataset_path)
        os.mkdir(dataset_path)
    return dataset_path


def create_model_dir_if_not_exist_and_get_path():
    root = expanduser("~")
    dataset_path = root + "/nlu_test_models/"
    if not os.path.exists(dataset_path):
        print("Creating dir", dataset_path)
        os.mkdir(dataset_path)
    return dataset_path


def create_path_if_not_exist(path):
    # Check if dir exists, if not create it
    import os

    if not os.path.exists(path):
        print("Creating dir", path)
        os.mkdir(path)


def model_and_output_levels_test(nlu_ref, lang, test_group=None, output_levels=None, input_data_type='generic',
                                 library='open_source'):
    from johnsnowlabs import nlp
    import tests.secrets as secrets
    if library == 'open_source':
        nlp.start()
    elif library == 'healthcare':
        nlp.start(json_license_path=secrets.JSON_LIC_PATH)
    elif library == 'ocr':
        nlp.start(json_license_path=secrets.JSON_LIC_PATH, visual=True)
    else:
        raise Exception(f'Library {library} is not supported')

    if not output_levels:
        # default everything except relation. Add it manually for RE models
        output_levels = ['entities', 'tokens', 'embeddings', 'document']
    for output_level in output_levels:
        model_test(nlu_ref, output_level=output_level, lang=lang, test_group=test_group,
                   input_data_type=input_data_type)


def model_test(nlu_ref, output_level=None, drop_irrelevant_cols=False, metadata=True, positions=True,
               test_group=None,
               lang='en',
               input_data_type='generic'):
    print(f'Testing Model {nlu_ref} with output_level={output_level} test_group={test_group}')
    pipe = nlu.load(nlu_ref, verbose=True)
    data = get_test_data(lang, input_data_type=input_data_type)

    df = pipe.predict(data, output_level=output_level,
                      drop_irrelevant_cols=drop_irrelevant_cols, metadata=metadata,
                      positions=positions)
    log_and_validate(df,test_group)

    if isinstance(data, list):
        df = pipe.predict(data[0], output_level=output_level,
                          drop_irrelevant_cols=drop_irrelevant_cols, metadata=metadata,
                          positions=positions)
        log_and_validate(df,test_group)


def validate_predictions(df):
    # TODO
    return True
