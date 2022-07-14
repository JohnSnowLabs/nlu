import numpy as np
import unittest

import pandas as pd

import nlu
import sys
import sparknlp

from nlu import NLP_FEATURES
from nlu.pipe.utils.data_conversion_utils import DataConversionUtils


class SpanQuestionConversionTestCase(unittest.TestCase):
    q = 'What is my name?'
    c = 'Heisenberg was a German theoretical physicist and oen of the key pioneers of quantum mechanics.'
    spark = sparknlp.start()

    def validate_conversion_result(self, spark_df):
        spark_df.show()
        self.assertTrue(NLP_FEATURES.RAW_QUESTION in spark_df.columns)
        self.assertTrue(NLP_FEATURES.RAW_QUESTION_CONTEXT in spark_df.columns)

    def test_str_conversion(self):
        data = f'{self.q}|||{self.c}'
        self.validate_conversion_result(DataConversionUtils.question_str_to_sdf(data, self.spark)[0])

    def test_str_iterable_conversion(self):
        data = f'{self.q}|||{self.c}'
        data = [data, data, data, data, data]
        self.validate_conversion_result(DataConversionUtils.question_str_iterable_to_sdf(data, self.spark)[0])
        ps = pd.Series(data)
        self.validate_conversion_result(DataConversionUtils.question_str_iterable_to_sdf(ps, self.spark)[0])

    def test_tuple_conversion(self):
        data = (self.q, self.c)
        self.validate_conversion_result(DataConversionUtils.question_tuple_to_sdf(data, self.spark)[0])

    def test_tuple_iterable_conversion(self):
        data = (self.q, self.c)
        data = [data, data, data, data, data]
        self.validate_conversion_result(DataConversionUtils.question_tuple_iterable_to_sdf(data, self.spark)[0])

        ps = pd.Series(data)
        self.validate_conversion_result(DataConversionUtils.question_tuple_iterable_to_sdf(ps, self.spark)[0])


    def test_sdf_conversion(self):
        data = pd.DataFrame({'question': [self.q],
                             'context': [self.c]
                             })
        data = self.spark.createDataFrame(data)

        self.validate_conversion_result(DataConversionUtils.question_sdf_to_sdf(data, self.spark)[0])
        data = pd.DataFrame({'c1': [self.q],
                             'c2': [self.c]
                             })
        data = self.spark.createDataFrame(data)

        self.validate_conversion_result(DataConversionUtils.question_sdf_to_sdf(data, self.spark)[0])

    def test_pdf_conversion(self):
        data = pd.DataFrame({'question': [self.q],
                             'context': [self.c]
                             })
        self.validate_conversion_result(DataConversionUtils.question_pdf_to_sdf(data, self.spark)[0])
        data = pd.DataFrame({'c1': [self.q],
                             'c2': [self.c]
                             })
        self.validate_conversion_result(DataConversionUtils.question_pdf_to_sdf(data, self.spark)[0])


if __name__ == '__main__':
    unittest.main()
