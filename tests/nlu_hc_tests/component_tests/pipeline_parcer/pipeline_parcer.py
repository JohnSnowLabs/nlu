import os
import sys
import json
sys.path.append(os.getcwd())
import unittest
import nlu

os.environ["PYTHONPATH"] = "F:/Work/repos/nlu"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import sparknlp
import sparknlp_jsl

from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

# params = {"spark.driver.memory":"24G",
#           "spark.kryoserializer.buffer.max":"2000M",
#           "spark.driver.maxResultSize":"20000M"}
with open('../../license.json') as f:
    license_keys = json.load(f)

# Defining license key-value pairs as local variables
locals().update(license_keys)
os.environ.update(license_keys)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

spark = sparknlp_jsl.start(secret = SECRET)

spark.sparkContext.setLogLevel("ERROR")

print ("Spark NLP Version :", sparknlp.version())
print ("Spark NLP_JSL Version :", sparknlp_jsl.version())

print ("\n\nspark session:", spark)
class DeidentificationTests(unittest.TestCase):
    def test_deidentification(self):
        res = nlu.load("en.de_identify").predict(
            "DR Johnson administerd to the patient Peter Parker last week 30 MG of penicilin",
            drop_irrelevant_cols=False,
            parser_output=True
        )

        print(res)
if __name__ == "__main__":
    DeidentificationTests().test_deidentification()

#scenario - 1
# column_maps = pipe.get_tracer()
# pipe.predict(parser_config=column_maps)

#scenario - 2
#pipe.predict(parser_output=True)