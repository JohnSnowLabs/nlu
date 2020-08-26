import unittest
import nlu
import pandas as pd
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_set_parameters(self):
        
        configs = {}
        configs['match.entities'] = ['dog', 'cat','sami']
        configs['sentiment.CaseSensitive'] = True
        
        predictions = nlu.load('sentiment').predict('That guy sami stinks')
    
    
if __name__ == '__main__':
    MyTestCase().test_entities_config()
