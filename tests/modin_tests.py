import unittest

import modin.pandas as mpd
import pandas as pd
import pyspark

import nlu


class ModinTests(unittest.TestCase):
    def test_modin(self):
        # ## works with RAY and DASK backends
        data = {
            "text": [
                "This day sucks but tomorrow will be better ! ",
                "I love this day",
                "I dont like Sami",
            ]
        }
        mdf = mpd.DataFrame(data)
        res = nlu.load("sentiment").predict(mdf)

        print(res)
        self.assertTrue(type(res) == mpd.DataFrame)
        print(data)
        pdf = pd.DataFrame(data)
        print(pdf)
        res = nlu.load("sentiment").predict(pdf)
        print(res)
        self.assertTrue(type(res) == pd.DataFrame)

        print("TESTING SDF")
        sdf = nlu.spark.createDataFrame(pdf)
        res = nlu.load("sentiment", verbose=True).predict(sdf)
        self.assertTrue(type(res) == pyspark.sql.dataframe.DataFrame)

        res.show()


if __name__ == "__main__":
    ModinTests().test_entities_config()
