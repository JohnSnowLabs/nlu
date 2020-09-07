import unittest
import nlu
import pandas as pd
import numpy as np


class TextPandasIndexReUsing(unittest.TestCase):
    def test_range_index(self):

        data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami'], "sentiment_label": [1, 1, 0]}
        text_df = pd.DataFrame(data)
        pipe =  nlu.load('sentiment').predict(text_df)
        print(text_df)
        
if __name__ == '__main__':
    TextPandasIndexReUsing().test_entities_config()
