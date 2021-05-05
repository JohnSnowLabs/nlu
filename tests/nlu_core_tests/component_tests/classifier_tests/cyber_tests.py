

import unittest
from nlu import *
class TestCyber(unittest.TestCase):


    def test_pos_train_bug(self):

        import nlu
        # load a trainable pipeline by specifying the train. prefix  and fit it on a datset with label and text columns
        # Since there are no
        train_path = '/home/ckl/Documents/freelance/jsl/nlu/nlu4realgit/UD_French-GSD_2.3.txt'
        trainable_pipe = nlu.load('train.pos')
        fitted_pipe = trainable_pipe.fit(dataset_path=train_path)

        # predict with the trainable pipeline on dataset and get predictions
        preds = fitted_pipe.predict('Donald Trump and Angela Merkel dont share many oppinions')
        preds

def test_cyber_model(self):
        import pandas as pd
        pipe = nlu.load('sentiment',verbose=True)
        df = pipe.predict(['Peter love pancaces. I hate Mondays', 'I love Fridays'], output_level='token',drop_irrelevant_cols=False, metadata=True, )
        for c in df.columns: print(df[c])

if __name__ == '__main__':
    unittest.main()

