"""Get data into JVM for prediction and out again as Spark Dataframe"""
import logging

from nlu.universe.feature_universes import NLP_FEATURES

logger = logging.getLogger('nlu')
import pyspark
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
import pandas as pd
from pyspark.sql.types import StringType, StructType, StructField


class DataConversionUtils:
    # Modin aswell but optional, so we dont import the type yet
    supported_types = [pyspark.sql.DataFrame, pd.DataFrame, pd.Series, np.ndarray]

    @staticmethod
    def except_text_col_not_found(cols):
        raise ValueError(
            f'Could not find column named "text" in input Pandas Dataframe. Please ensure one column named such exists. Columns in DF are : {cols} ')

    @staticmethod
    def except_invalid_question_data_format(cols):
        raise ValueError(
            f'You input data format is invalid for question answering with span classification.'
            f'Make sure you have at least 2 columns in you dataset, named context/question  for pandas Dataframes'
            f'For Strings/Iterables/Tuples make sure to use the format `question|||context` or (question,context) ' )

    @staticmethod
    def sdf_to_sdf(data, spark_sess, raw_text_column='text'):
        """No casting, Spark to Spark. Just add index col"""
        logger.info(f"Casting Spark DF to Spark DF")
        output_datatype = 'spark'
        data = data.withColumn('origin_index', monotonically_increasing_id().alias('origin_index'))
        stranger_features = []
        if raw_text_column in data.columns:
            # store all stranger features
            if len(data.columns) > 1:
                stranger_features = list(set(data.columns) - set(raw_text_column))
            else:
                DataConversionUtils.except_text_col_not_found(data.columns)
        return data, stranger_features, output_datatype

    @staticmethod
    def question_sdf_to_sdf(data, spark_sess):
        """Casting question pandas to spark and add index col"""
        logger.info(f"Casting Pandas DF to Spark DF")
        output_datatype = 'spark'
        if NLP_FEATURES.RAW_QUESTION not in data.columns or NLP_FEATURES.RAW_QUESTION_CONTEXT not in data.columns:
            if len(data.columns) < 2:
                DataConversionUtils.except_invalid_question_data_format(data)
            data = data.withColumnRenamed(data.columns[0], NLP_FEATURES.RAW_QUESTION) \
                .withColumnRenamed(data.columns[1], NLP_FEATURES.RAW_QUESTION_CONTEXT)

        data = data.withColumn('origin_index', monotonically_increasing_id().alias('origin_index'))
        # make  Nans to None, or spark will crash
        stranger_features = list(set(data.columns) - {NLP_FEATURES.RAW_QUESTION, NLP_FEATURES.RAW_QUESTION_CONTEXT})
        return data, stranger_features, output_datatype

    @staticmethod
    def question_str_to_sdf(data, spark_sess):
        """Casting str  to spark and add index col. This is a bit inefficient. Casting follow  # inefficient, str->pd->spark->pd , we can could first pd"""
        output_datatype = 'string'
        if '|||' not in data:
            DataConversionUtils.except_invalid_question_data_format(data)
        question, context = data.split('|||')
        sdf = spark_sess.createDataFrame(pd.DataFrame({NLP_FEATURES.RAW_QUESTION: question,
                                                       NLP_FEATURES.RAW_QUESTION_CONTEXT: context,
                                                       'origin_index': [0]}, index=[0]))
        return sdf, [], output_datatype

    @staticmethod
    def question_tuple_to_sdf(data, spark_sess):
        """Casting str  to spark and add index col. This is a bit inefficient. Casting follow  # inefficient, str->pd->spark->pd , we can could first pd"""
        output_datatype = 'string'
        question, context = data[0], data[1]
        sdf = spark_sess.createDataFrame(pd.DataFrame({NLP_FEATURES.RAW_QUESTION: question,
                                                       NLP_FEATURES.RAW_QUESTION_CONTEXT: context,
                                                       'origin_index': [0]}, index=[0]))
        return sdf, [], output_datatype

    @staticmethod
    def question_tuple_iterable_to_sdf(data, spark_sess):
        """Casting str  to spark and add index col. This is a bit inefficient. Casting follow  # inefficient, str->pd->spark->pd , we can could first pd"""
        output_datatype = 'string'

        if len(data) == 0:
            DataConversionUtils.except_invalid_question_data_format(data)
        if len(data[0]) != 2:
            DataConversionUtils.except_invalid_question_data_format(data)

        question, context = zip(*[(d[0], d[1]) for d in data])

        sdf = spark_sess.createDataFrame(pd.DataFrame({NLP_FEATURES.RAW_QUESTION: question,
                                                       NLP_FEATURES.RAW_QUESTION_CONTEXT: context,
                                                       'origin_index': [0]}, index=list(range(len(question)))))
        return sdf, [], output_datatype

    @staticmethod
    def question_str_iterable_to_sdf(data, spark_sess):
        """Casting str  to spark and add index col. This is a bit inefficient. Casting follow  # inefficient, str->pd->spark->pd , we can could first pd"""
        output_datatype = 'string'
        if len(data) == 0:
            DataConversionUtils.except_invalid_question_data_format(data)
        if '|||' not in data[0]:
            DataConversionUtils.except_invalid_question_data_format(data)
        question, context = zip(*[d.split('|||') for d in data])
        sdf = spark_sess.createDataFrame(pd.DataFrame({NLP_FEATURES.RAW_QUESTION: question,
                                                       NLP_FEATURES.RAW_QUESTION_CONTEXT: context,
                                                       'origin_index': list(range(len(question)))}))
        return sdf, [], output_datatype

    @staticmethod
    def pdf_to_sdf(data, spark_sess, raw_text_column='text'):
        """Casting pandas to spark and add index col"""
        logger.info(f"Casting Pandas DF to Spark DF")
        output_datatype = 'pandas'
        stranger_features = []
        sdf = None
        # set first col as text column if there is none
        if raw_text_column not in data.columns: data.rename(columns={data.columns[0]: 'text'}, inplace=True)
        data['origin_index'] = data.index
        if raw_text_column in data.columns:
            if len(data.columns) > 1:
                # make  Nans to None, or spark will crash
                data = data.where(pd.notnull(data), None)
                data = data.dropna(axis=1, how='all')
                stranger_features = list(set(data.columns) - set(raw_text_column))
            sdf = spark_sess.createDataFrame(data)
        else:
            DataConversionUtils.except_text_col_not_found(data.columns)
        return sdf, stranger_features, output_datatype

    @staticmethod
    def question_pdf_to_sdf(data, spark_sess):
        """Casting question pandas to spark and add index col"""
        logger.info(f"Casting Pandas DF to Spark DF")
        output_datatype = 'pandas'
        if NLP_FEATURES.RAW_QUESTION not in data.columns or NLP_FEATURES.RAW_QUESTION_CONTEXT not in data.columns:
            if len(data.columns) < 2:
                DataConversionUtils.except_invalid_question_data_format(data)
            data = data.rename(columns={
                data.columns[0]: NLP_FEATURES.RAW_QUESTION,
                data.columns[1]: NLP_FEATURES.RAW_QUESTION_CONTEXT,
            })

        data['origin_index'] = data.index
        # make  Nans to None, or spark will crash
        data = data.where(pd.notnull(data), None)
        data = data.dropna(axis=1, how='all')
        stranger_features = list(set(data.columns) - {NLP_FEATURES.RAW_QUESTION, NLP_FEATURES.RAW_QUESTION_CONTEXT})
        sdf = spark_sess.createDataFrame(data)
        return sdf, stranger_features, output_datatype

    @staticmethod
    def pds_to_sdf(data, spark_sess, raw_text_column='text'):
        """Casting pandas series to spark and add index col.  # for df['text'] colum/series passing casting follows pseries->pdf->spark->pd """
        logger.info(f"Casting Pandas Series to Spark DF")

        output_datatype = 'pandas_series'
        sdf = None
        schema = StructType([StructField(raw_text_column, StringType(), True)])
        data = pd.DataFrame(data).dropna(axis=1, how='all')
        # If series from a column is passed, its column name will be reused.
        if raw_text_column not in data.columns and len(data.columns) == 1:
            data[raw_text_column] = data[data.columns[0]]
        else:
            logger.info(
                f'INFO: NLU will assume {data.columns[0]} as label column since default text column could not be find')
            data[raw_text_column] = data[data.columns[0]]
        data['origin_index'] = data.index
        if raw_text_column in data.columns:
            sdf = spark_sess.createDataFrame(pd.DataFrame(data[raw_text_column]), schema=schema)
        else:
            DataConversionUtils.except_text_col_not_found(data.columns)
        if 'origin_index' not in sdf.columns:
            sdf = sdf.withColumn('origin_index', monotonically_increasing_id().alias('origin_index'))
        return sdf, [], output_datatype

    @staticmethod
    def np_to_sdf(data, spark_sess, raw_text_column='text'):
        """Casting numpy array to spark and add index col. This is a bit inefficient. Casting follow  np->pd->spark->pd. We could cut out the first pd step   """
        logger.info(f"Casting Numpy Array to Spark DF")
        output_datatype = 'numpy_array'
        if len(data.shape) != 1: ValueError(
            f"Exception : Input numpy array must be 1 Dimensional for prediction.. Input data shape is{data.shape}")
        sdf = spark_sess.createDataFrame(pd.DataFrame({raw_text_column: data, 'origin_index': list(range(len(data)))}))
        return sdf, [], output_datatype

    @staticmethod
    def str_to_sdf(data, spark_sess, raw_text_column='text'):
        """Casting str  to spark and add index col. This is a bit inefficient. Casting follow  # inefficient, str->pd->spark->pd , we can could first pd"""
        logger.info(f"Casting String to Spark DF")
        output_datatype = 'string'
        sdf = spark_sess.createDataFrame(pd.DataFrame({raw_text_column: data, 'origin_index': [0]}, index=[0]))
        return sdf, [], output_datatype

    @staticmethod
    def str_list_to_sdf(data, spark_sess, raw_text_column='text'):
        """Casting str list  to spark and add index col. This is a bit inefficient. Casting follow  # # inefficient, list->pd->spark->pd , we can could first pd"""
        logger.info(f"Casting String List to Spark DF")
        output_datatype = 'string_list'
        if all(type(elem) == str for elem in data):
            sdf = spark_sess.createDataFrame(
                pd.DataFrame({raw_text_column: pd.Series(data), 'origin_index': list(range(len(data)))}))
        else:
            ValueError("Exception: Not all elements in input list are of type string.")
        return sdf, [], output_datatype

    @staticmethod
    def fallback_modin_to_sdf(data, spark_sess, raw_text_column='text'):
        """Casting potential Modin data to spark and add index col. # Modin tests, This could crash if Modin not installed """
        logger.info(f"Casting Modin DF to Spark DF")
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
            else:
                DataConversionUtils.except_text_col_not_found(data.columns)
            if isinstance(data, mpd.Series):
                output_datatype = 'modin_series'
                data = pd.Series(data.to_dict())  # create pandas to support type inference
                data = pd.DataFrame(data).dropna(axis=1, how='all')
                data['origin_index'] = data.index
                index_provided = True
                if raw_text_column in data.columns:
                    sdf = spark_sess.createDataFrame(data[['text']])
                else:
                    DataConversionUtils.except_text_col_not_found(data.columns)
        except:
            print(
                "If you use Modin, make sure you have installed 'pip install modin[ray]' or 'pip install modin[dask]' backend for Modin ")
        return sdf, [], output_datatype

    @staticmethod
    def to_spark_df(data, spark_sess, raw_text_column='text', is_span_data=False):
        """Convert supported datatypes to SparkDF and extract extra data for prediction later on."""
        if is_span_data:
            try:
                if isinstance(data, pyspark.sql.dataframe.DataFrame):
                    return DataConversionUtils.question_sdf_to_sdf(data, spark_sess)
                elif isinstance(data, pd.DataFrame):
                    return DataConversionUtils.question_pdf_to_sdf(data, spark_sess)
                elif isinstance(data, tuple):
                    return DataConversionUtils.question_tuple_to_sdf(data, spark_sess)
                elif isinstance(data, str):
                    return DataConversionUtils.question_str_to_sdf(data, spark_sess)
                elif isinstance(data, (list, pd.Series, np.ndarray)):
                    if isinstance(data[0], tuple):
                        return DataConversionUtils.question_tuple_iterable_to_sdf(data, spark_sess)
                    elif isinstance(data[0], str):
                        return DataConversionUtils.question_str_iterable_to_sdf(data, spark_sess)
            except:
                ValueError("Data could not be converted to Spark Dataframe for internal conversion.")
        else:
            try:
                if isinstance(data, pyspark.sql.dataframe.DataFrame):
                    return DataConversionUtils.sdf_to_sdf(data, spark_sess, raw_text_column)
                elif isinstance(data, pd.DataFrame):
                    return DataConversionUtils.pdf_to_sdf(data, spark_sess, raw_text_column)
                elif isinstance(data, pd.Series):
                    return DataConversionUtils.pds_to_sdf(data, spark_sess, raw_text_column)
                elif isinstance(data, np.ndarray):
                    return DataConversionUtils.np_to_sdf(data, spark_sess, raw_text_column)
                elif isinstance(data, str):
                    return DataConversionUtils.str_to_sdf(data, spark_sess, raw_text_column)
                elif isinstance(data, list):
                    return DataConversionUtils.str_list_to_sdf(data, spark_sess, raw_text_column)
                else:
                    return DataConversionUtils.fallback_modin_to_sdf(data, spark_sess, raw_text_column)
            except:
                ValueError("Data could not be converted to Spark Dataframe for internal conversion.")
        raise TypeError(f"Invalid datatype = {type(data)}")

    @staticmethod
    def str_to_pdf(data, raw_text_column):
        logger.info(f"Casting String to Pandas DF")
        return pd.DataFrame({raw_text_column: [data]}).reset_index().rename(
            columns={'index': 'origin_index'}), [], 'string'

    @staticmethod
    def str_list_to_pdf(data, raw_text_column):
        logger.info(f"Casting String List to Pandas DF")
        return pd.DataFrame({raw_text_column: data}).reset_index().rename(
            columns={'index': 'origin_index'}), [], 'string_list'

    @staticmethod
    def np_to_pdf(data, raw_text_column):
        logger.info(f"Casting Numpy Array to Pandas DF")
        return pd.DataFrame({raw_text_column: data}).reset_index().rename(
            columns={'index': 'origin_index'}), [], 'string_list'

    @staticmethod
    def pds_to_pdf(data, raw_text_column):
        return pd.DataFrame({raw_text_column: data}).reset_index().rename(
            columns={'index': 'origin_index'}), [], 'string_list'

    @staticmethod
    def pdf_to_pdf(data, raw_text_column):
        logger.info(f"Casting Pandas DF to Pandas DF")
        data = data.reset_index().rename(columns={'index': 'origin_index'})
        stranger_features = list(data.columns)
        if raw_text_column not in stranger_features:
            print(f"Could not find {raw_text_column} col in df. Using {stranger_features[0]} col istead")
            data = data.reset_index().rename(columns={stranger_features[0]: raw_text_column})
        stranger_features.remove('text')
        stranger_features.remove('origin_index')

        return data, stranger_features, 'pandas'

    @staticmethod
    def sdf_to_pdf(data, raw_text_column):
        logger.info(f"Casting Spark DF to Pandas DF")
        data = data.toPandas().reset_index().rename(columns={'index': 'origin_index'})
        stranger_features = list(data.columns)
        if raw_text_column not in stranger_features:
            print(f"Could not find {raw_text_column} col in df. Using {stranger_features[0]} col istead")
            data = data.reset_index().rename(columns={stranger_features[0]: raw_text_column})
        stranger_features.remove('text')
        stranger_features.remove('origin_index')

        return data, stranger_features, 'spark'

    @staticmethod
    def to_pandas_df(data, raw_text_column='text'):
        """
        Convert data to LihgtPipeline Compatible Format, which is np.array[str], list[str] and str  but we need list anyways later.
        So we create here a pd.Dataframe with a TEXT col if not already given
        Convert supported datatypes to Pandas and extract extra data for prediction later on.

        """
        try:
            if isinstance(data, pyspark.sql.dataframe.DataFrame):
                return DataConversionUtils.pdf_to_pdf(data, raw_text_column)
            elif isinstance(data, pd.DataFrame):
                return DataConversionUtils.pdf_to_pdf(data, raw_text_column)
            elif isinstance(data, pd.Series):
                return DataConversionUtils.pds_to_pdf(data, raw_text_column)
            elif isinstance(data, np.ndarray):
                return DataConversionUtils.np_to_pdf(data, raw_text_column)
            elif isinstance(data, str):
                return DataConversionUtils.str_to_pdf(data, raw_text_column)
            elif isinstance(data, list):
                return DataConversionUtils.str_list_to_pdf(data, raw_text_column)
            else:
                return DataConversionUtils.fallback_modin_to_pdf(data, raw_text_column)
        except:
            ValueError("Data could not be converted to Spark Dataframe for internal conversion.")

    @staticmethod
    def size_of(data):
        """
        Convert data to LihgtPipeline Compatible Format, which is np.array[str], list[str] and str  but we need list anyways later.
        So we create here a pd.Dataframe with a TEXT col if not already given
        Convert supported datatypes to Pandas and extract extra data for prediction later on.

        """
        if isinstance(data, pyspark.sql.dataframe.DataFrame):
            return data.count()
        elif isinstance(data, pd.DataFrame):
            return data.shape[0]
        elif isinstance(data, pd.Series):
            return data.shape[0]
        elif isinstance(data, np.ndarray):
            return data.shape[0]
        elif isinstance(data, str):
            return 1
        elif isinstance(data, list):
            return len(data)
        else:
            return len(data)
