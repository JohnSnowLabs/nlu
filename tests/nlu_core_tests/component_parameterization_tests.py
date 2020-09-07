import unittest
import nlu
import pandas as pd
import numpy as np


class TestParameterization(unittest.TestCase):
    def test_set_parameters(self):


        pipe =  nlu.load('sentiment')
        print(pipe.keys())    
        pipe.print_info()
    
if __name__ == '__main__':
    TestParameterization().test_entities_config()
