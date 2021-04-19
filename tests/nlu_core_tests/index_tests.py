import unittest
import nlu
import pandas as pd
import numpy as np


class TextPandasIndexReUsing(unittest.TestCase):
    def test_range_index(self):

        data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami'], "some feature": [1, 1, 0]}
        text_df = pd.DataFrame(data)
        df =  nlu.load('sentiment',verbose=True).predict(text_df,output_level='document')
        for c in df.columns: print(df[c])

        
if __name__ == '__main__':
    TextPandasIndexReUsing().test_entities_config()
