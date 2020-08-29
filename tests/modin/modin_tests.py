import unittest
import pandas as pd
import numpy as np
import modin.pandas as mpd
import numpy as np
import nlu
import sparknlp
import pyspark

class MyTestCase(unittest.TestCase):
    def test_print_pipe_info(self):
        # ## works with RAY and DASK backends
        data = {"text": ['This day sucks but tomorrow will be better ! ', 'I love this day', 'I dont like Sami']}
        mdf = mpd.DataFrame(data)
        res = nlu.load('sentiment').predict(mdf)

        print(res)
        self.assertTrue(type(res) == mpd.DataFrame)
        pdf = pd.DataFrame(data)
        res = nlu.load('sentiment').predict(pdf)
        print(res)
        self.assertTrue(type(res) == pd.DataFrame)


        print('TESTING SDF')
        sdf = nlu.spark.createDataFrame(pdf)
        res = nlu.load('sentiment', verbose=True).predict(sdf)
        self.assertTrue(type(res) == pyspark.sql.dataframe.DataFrame)
    
        res.show()
if __name__ == '__main__':
    MyTestCase().test_entities_config()
