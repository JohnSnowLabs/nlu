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
