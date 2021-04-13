import nlu
import pandas as pd
import sparknlp

spark = sparknlp.start()


def get_sample_pdf():
    data = {"text": ['This day sucks but tomorrow will be better ! ', 'I love this day', 'I dont like Sami']}
    text_df = pd.DataFrame(data)
    return text_df


def get_sample_pdf_with_labels():
    data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami'], "sentiment_label": [1, 1, 0]}
    text_df = pd.DataFrame(data)
    return text_df


def get_sample_sdf():
    nlu.spark = sparknlp.start()
    nlu.spark_started = True
    return nlu.spark.createDataFrame(get_sample_pdf())


def get_sample_pdf_with_extra_cols():
    data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami'], "random_feature1": [1, 1, 0], "random_feature2": ['d','a' , '3']}
    text_df = pd.DataFrame(data)
    return text_df

def get_sample_pdf_with_no_text_col():
    data = {"schmext": ['This day sucks', 'I love this day', 'I dont like Sami'], "random_feature1": [1, 1, 0], "random_feature2": ['d','a' , '3']}
    text_df = pd.DataFrame(data)
    return text_df

def get_sample_spark_dataframe():
    data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami'], "random_feature1": [1, 1, 0], "random_feature2": ['d','a' , '3']}
    text_df = pd.DataFrame(data)
    return text_df

def get_sample_pdf_with_extra_cols_and_entities():
    data = {"text": ['Pater says this day sucks. He lives in America. He likes Angela Merkel from Germany', 'I love burgers from Burger King', 'I dont like Sami, he lives in Asia'], "random_feature1": [1, 1, 0], "random_feature2": ['d','a' , '3']}
    text_df = pd.DataFrame(data)
    return text_df


from os.path import expanduser

import os

def download_dataset(data_url,output_file_name,output_folder,):
    import urllib.request
    import os
    download_path = create_dataset_dir_if_not_exist_and_get_path() + output_folder + output_file_name

    #Check if dir exists, if not create it
    # create_path_if_not_exist(data_dir )
    create_path_if_not_exist(create_dataset_dir_if_not_exist_and_get_path() + output_folder)


    from pathlib import Path
    #Check if file exists, if not download it
    if not Path(download_path).is_file():
        urllib.request.urlretrieve(data_url, download_path )

    print('Downloaded dataset to ',download_path)
    return download_path


def create_dataset_dir_if_not_exist_and_get_path():
    root = expanduser('~')
    dataset_path = root + '/nlu_test_datasets/'
    if not os.path.exists(dataset_path):
        print('Creating dir',dataset_path)
        os.mkdir(dataset_path)
    return dataset_path

def create_model_dir_if_not_exist_and_get_path():
    root = expanduser('~')
    dataset_path = root + '/nlu_test_models/'
    if not os.path.exists(dataset_path):
        print('Creating dir',dataset_path)
        os.mkdir(dataset_path)
    return dataset_path


def create_path_if_not_exist(path):
    #Check if dir exists, if not create it
    import os
    if not os.path.exists(path):
        print('Creating dir',path)
        os.mkdir(path)