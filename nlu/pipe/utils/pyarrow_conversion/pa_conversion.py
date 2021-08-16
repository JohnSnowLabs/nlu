import logging
logger = logging.getLogger('nlu')
from pyspark.sql.dataframe import DataFrame
import numpy as np
import pandas as pd
from pyspark.sql.types import ByteType, ShortType, IntegerType, LongType, FloatType, \
    DoubleType, BooleanType, MapType, TimestampType, StructType, DataType

from pyspark.sql.pandas.types import _check_series_localize_timestamps, \
    _convert_map_items_to_dict
from pyspark.sql.pandas.utils import require_minimum_pandas_version

import pyarrow

class PaConversionUtils():
    @staticmethod
    def convert_via_pyarrow(sdf:DataFrame   ) -> pd.DataFrame:
        """Convert a Spark Dataframe to a pandas Dataframe using PyArrow shared memory blocks between Spark and Pandas backends.

        Args:
            sdf:DataFrame
        """
        require_minimum_pandas_version()
        timezone = sdf.sql_ctx._conf.sessionLocalTimeZone()
        # Rename columns to avoid duplicated column names.
        tmp_column_names = ['col_{}'.format(i) for i in range(len(sdf.columns))]
        batches = sdf.toDF(*tmp_column_names)._collect_as_arrow()
        if len(batches) > 0:
            table = pyarrow.Table.from_batches(batches)
            # Pandas DataFrame created from PyArrow uses datetime64[ns] for date type
            # values, but we should use datetime.date to match the behavior with when
            # Arrow optimization is disabled.
            pdf = table.to_pandas(date_as_object=True)
            # Rename back to the original column names.
            pdf.columns = sdf.columns
            for field in sdf.schema:
                if isinstance(field.dataType, TimestampType):
                    pdf[field.name] = \
                        _check_series_localize_timestamps(pdf[field.name], timezone)
                elif isinstance(field.dataType, MapType):
                    pdf[field.name] = \
                        _convert_map_items_to_dict(pdf[field.name])
            return pdf
        else:return pd.DataFrame.from_records([], columns=sdf.columns)

