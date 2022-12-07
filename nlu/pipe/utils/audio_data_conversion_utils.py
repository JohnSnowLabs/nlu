from pyspark.sql.types import *
import glob
import logging
import os
from typing import List
import pyspark
import numpy as np
import pandas as pd

logger = logging.getLogger('nlu')


class AudioDataConversionUtils:
    """Validate Audio Data Files and Create Spark DataFrames from them"""

    @staticmethod
    def validate_paths(data):
        """Validate for input data that it contains a path pointing to file or folder of audio fila readable with librosa"""
        if isinstance(data, List):
            return AudioDataConversionUtils.check_iterable_paths_are_valid(data)
        if isinstance(data, str):
            return AudioDataConversionUtils.check_iterable_paths_are_valid([data])
        if isinstance(data, pd.DataFrame):
            return 'path' in data.columns
        if isinstance(data, pd.Series):
            return 'path' in data.name
        if isinstance(data, pyspark.sql.dataframe.DataFrame):
            return 'path' in data.columns
        if isinstance(data, np.ndarray):
            return AudioDataConversionUtils.check_iterable_paths_are_valid(data)

    @staticmethod
    def check_iterable_paths_are_valid(iterable_paths):
        """Validate for iterable data input if all elements point to file or jsl_folder"""
        paths_validness = []
        for p in iterable_paths:
            if os.path.isdir(p) or os.path.isfile(p):
                paths_validness.append(True)
            else:
                print(f'Warning : Invalid path for jsl_folder or file in input. Could validate path.\n'
                      f'NLU will try and ignore this issue but you might run into errors.\n'
                      f'Please make sure all paths are valid\n')
                print(f'For path = {p}')
                paths_validness.append(False)
        return all(paths_validness)

    @staticmethod
    def check_all_paths_point_to_accepted_file_type(paths, file_types):
        """Validate that all paths point to a file type defined by file_types"""
        pass

    @staticmethod
    def data_to_spark_audio_df(data, sample_rate, spark):
        import librosa
        import typing
        if isinstance(data, str):
            data, _ = librosa.load(data, sr=sample_rate)
            # let's convert them to floats
            df = pd.DataFrame({
                # data is an List[List[Float]]
                "raw_audio": data.toList(),
                "sampling_rate": [sample_rate]
            })
        elif isinstance(data, typing.Iterable):
            data_ = []
            for d in data:
                data, _ = librosa.load(d, sr=sample_rate)
                data_.append(data.tolist())
            df = pd.DataFrame({
                # data is an List[List[Float]]
                "raw_audio": data_,
                "sampling_rate": [sample_rate] * len(data_)
            })
        schema = StructType([StructField("raw_audio", ArrayType(FloatType())),
                             StructField("sampling_rate", LongType())])
        data = spark.createDataFrame(df, schema)
        return data

    @staticmethod
    def glob_files_of_accepted_type(paths, file_types):
        """Get all paths which point to correct file types from iterable paths which can contain file and jsl_folder paths
        1. paths point to a file which is suffixed with one of the accepted file_types, i.e. path/to/file.type
        2. path points to a jsl_folder, in this case jsl_folder is recursively searched for valid files and accepted paths will be in return result
        """
        accepted_file_paths = []
        for p in paths:
            for t in file_types:
                t = t.lower()
                if os.path.isfile(p):
                    if p.lower().split('.')[-1] == t:
                        accepted_file_paths.append(p)
                elif os.path.isdir(p):
                    accepted_file_paths += glob.glob(p + f'/**/*.{t}', recursive=True)
                else:
                    print(f"Invalid path = {p} pointing neither to file or jsl_folder on this machine")
        return accepted_file_paths

    @staticmethod
    def extract_iterable_paths_from_data(data):
        """Extract an iterable object containing paths from input data"""
        if isinstance(data, List):
            return data
        if isinstance(data, str):
            return [data]
        if isinstance(data, pd.DataFrame):
            return list(data['path'].values)
        if isinstance(data, pd.Series):
            return list(data.values)
        if isinstance(data, pyspark.sql.dataframe.DataFrame):
            return [p['path'] for p in data.select('path').collect()]
        if isinstance(data, np.ndarray):
            return list(data)

    @staticmethod
    def get_accepted_audio_file_types(pipe):
        """Get all file types/suffixes that can be processed by the pipeline"""
        accepted_files = []
        for c in pipe.components:
            if c.applicable_file_types:
                accepted_files += c.applicable_file_types
        return list(set(accepted_files))
