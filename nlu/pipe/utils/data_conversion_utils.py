"""Get data into JVM for prediction and out again as Spark Dataframe"""
import logging
logger = logging.getLogger('nlu')
import pyspark
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
import pandas as pd
from pyspark.sql.types import StringType, StructType,StructField

class DataConversionUtils():
    # Modin aswell but optional, so we dont import the type yet
    supported_types = [pyspark.sql.DataFrame,pd.DataFrame, pd.Series,np.ndarray ]
    @staticmethod
    def except_text_col_not_found(cols):
        print(f'Could not find column named "text" in input Pandas Dataframe. Please ensure one column named such exists. Columns in DF are : {cols} ')
    @staticmethod
    def sdf_to_sdf(data,spark_sess,raw_text_column='text'):
        """No casting, Spark to Spark. Just add index col"""
        output_datatype = 'spark'
        data = data.withColumn('origin_index',monotonically_increasing_id().alias('origin_index'))
        stranger_features= []
        if raw_text_column in data.columns:
            # store all stranger features
            if len(data.columns) > 1:
                stranger_features = list(set(data.columns) - set(raw_text_column))
            else: DataConversionUtils.except_text_col_not_found(data.columns)
        return data, stranger_features, output_datatype

    @staticmethod
    def pdf_to_sdf(data,spark_sess,raw_text_column='text'):
        """Casting pandas to spark and add index col"""
        output_datatype = 'pandas'
        stranger_features= []
        sdf = None
        # set first col as text column if there is none
        if raw_text_column not in data.columns:data.rename(columns={data.columns[0]: 'text'}, inplace=True)
        data['origin_index'] = data.index
        if raw_text_column in data.columns:
            if len(data.columns) > 1:
                # make  Nans to None, or spark will crash
                data = data.where(pd.notnull(data), None)
                data = data.dropna(axis=1, how='all')
                stranger_features = list(set(data.columns) - set(raw_text_column))
            sdf = spark_sess.createDataFrame(data)
        else: DataConversionUtils.except_text_col_not_found(data.columns)
        return sdf, stranger_features, output_datatype

    @staticmethod
    def pds_to_sdf(data,spark_sess,raw_text_column='text'):
        """Casting pandas series to spark and add index col.  # for df['text'] colum/series passing casting follows pseries->pdf->spark->pd """
        output_datatype = 'pandas_series'
        sdf=None
        schema = StructType([StructField(raw_text_column,StringType(),True)])
        data = pd.DataFrame(data).dropna(axis=1, how='all')
        # If series from a column is passed, its column name will be reused.
        if raw_text_column not in data.columns and len(data.columns) == 1:
            data[raw_text_column] = data[data.columns[0]]
        else:
            logger.info(f'INFO: NLU will assume {data.columns[0]} as label column since default text column could not be find')
            data[raw_text_column] = data[data.columns[0]]
        data['origin_index'] = data.index
        if raw_text_column in data.columns:
            sdf = spark_sess.createDataFrame(pd.DataFrame(data[raw_text_column]),schema=schema)
        else: DataConversionUtils.except_text_col_not_found(data.columns)
        return sdf, [], output_datatype

    @staticmethod
    def np_to_sdf(data,spark_sess,raw_text_column='text'):
        """Casting numpy array to spark and add index col. This is a bit inefficient. Casting follow  np->pd->spark->pd. We could cut out the first pd step   """
        output_datatype = 'numpy_array'
        if len(data.shape) != 1: ValueError(f"Exception : Input numpy array must be 1 Dimensional for prediction.. Input data shape is{data.shape}")
        sdf = spark_sess.createDataFrame(pd.DataFrame({raw_text_column: data, 'origin_index': list(range(len(data)))}))
        return sdf, [], output_datatype

    @staticmethod
    def str_to_sdf(data,spark_sess,raw_text_column='text'):
        """Casting str  to spark and add index col. This is a bit inefficient. Casting follow  # inefficient, str->pd->spark->pd , we can could first pd"""
        output_datatype = 'string'
        sdf = spark_sess.createDataFrame(pd.DataFrame({raw_text_column: data, 'origin_index': [0]}, index=[0]))
        return sdf, [], output_datatype

    @staticmethod
    def str_list_to_sdf(data,spark_sess,raw_text_column='text'):
        """Casting str list  to spark and add index col. This is a bit inefficient. Casting follow  # # inefficient, list->pd->spark->pd , we can could first pd"""
        output_datatype = 'string_list'
        if all(type(elem) == str for elem in data):
            sdf = spark_sess.createDataFrame(pd.DataFrame({raw_text_column: pd.Series(data), 'origin_index': list(range(len(data)))}))
        else:
            ValueError("Exception: Not all elements in input list are of type string.")
        return sdf, [], output_datatype


    @staticmethod
    def fallback_modin_to_sdf(data,spark_sess,raw_text_column='text'):
        """Casting potential Modin data to spark and add index col. # Modin tests, This could crash if Modin not installed """
        sdf = None
        output_datatype = ''
        try:
            import modin.pandas as mpd
            if isinstance(data, mpd.DataFrame):
                data = pd.DataFrame(data.to_dict())  # create pandas to support type inference
                output_datatype = 'modin'
                data['origin_index'] = data.index
            if raw_text_column in data.columns:
                if len(data.columns) > 1:
                    data = data.where(pd.notnull(data), None)  # make  Nans to None, or spark will crash
                    data = data.dropna(axis=1, how='all')
                    stranger_features = list(set(data.columns) - set(raw_text_column))
                sdf = spark_sess.createDataFrame(data)
            else: DataConversionUtils.except_text_col_not_found(data.columns)
            if isinstance(data, mpd.Series):
                output_datatype = 'modin_series'
                data = pd.Series(data.to_dict())  # create pandas to support type inference
                data = pd.DataFrame(data).dropna(axis=1, how='all')
                data['origin_index'] = data.index
                index_provided = True
                if raw_text_column in data.columns:
                    sdf = spark_sess.createDataFrame(data[['text']])
                else: DataConversionUtils.except_text_col_not_found(data.columns)
        except: print("If you use Modin, make sure you have installed 'pip install modin[ray]' or 'pip install modin[dask]' backend for Modin ")
        return sdf, [], output_datatype


    @staticmethod
    def to_spark_df(data,spark_sess,raw_text_column='text') :
        """Convert supported datatypes to SparkDF and extract extra data for prediction later on."""
        try :
            if   isinstance(data,pyspark.sql.dataframe.DataFrame): return DataConversionUtils.sdf_to_sdf(data,spark_sess,raw_text_column)
            elif isinstance(data,pd.DataFrame):  return DataConversionUtils.pdf_to_sdf(data,spark_sess,raw_text_column)
            elif isinstance(data,pd.Series):     return DataConversionUtils.pds_to_sdf(data,spark_sess,raw_text_column)
            elif isinstance(data,np.ndarray):  return DataConversionUtils.np_to_sdf(data,spark_sess,raw_text_column)
            elif isinstance(data,str): return DataConversionUtils.str_to_sdf(data,spark_sess,raw_text_column)
            elif isinstance(data,list): return DataConversionUtils.str_list_to_sdf(data,spark_sess,raw_text_column)
            else: return DataConversionUtils.fallback_modin_to_sdf(data,spark_sess,raw_text_column)
        except : ValueError("Data could not be converted to Spark Dataframe for internal conversion.")