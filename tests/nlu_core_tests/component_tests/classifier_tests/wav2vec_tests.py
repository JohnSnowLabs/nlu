import unittest
import sparknlp
import librosa as librosa
from sparknlp.base import *
from sparknlp.annotator import *
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
import pyspark.sql.functions as F
import sparknlp
import sparknlp
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
import os


os.environ['PYSPARK_PYTHON'] = '/home/ckl/anaconda3/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/ckl/anaconda3/bin/python3'



class Wav2VecCase(unittest.TestCase):
    def test_wav2vec(self):
        import nlu
        p = nlu.load('en.wav2vec.wip',verbose=True)
        FILE_PATH = os.path.normpath(r"tests/datasets/audio/asr/ngm_12484_01067234848.wav")

        print("Got p ",p)
        df = p.predict(FILE_PATH)
        print(df)
        df = p.predict([FILE_PATH,FILE_PATH])
        print(df)

if __name__ == '__main__':
    unittest.main()
