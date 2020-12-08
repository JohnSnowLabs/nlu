

from sklearn.metrics import classification_report

import unittest
from nlu import *
class posTrainingTests(unittest.TestCase):

    def test_pos_training(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('pos classifier_dl bert') will only give trainable classifier dl
        #


        #pos datase
        train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/pos/UD_French-GSD_2.3.txt'
        # df_train = pd.read_csv(train_path,error_bad_lines=False)
        #convert int to str labels so our model predicts strings not numbers
        # df_train.dropna(inplace=True)

        pipe = nlu.load('train.pos',verbose=True)
        fitted_pipe = pipe.fit(dataset_path=train_path)
        # df = fitted_pipe.predict(' I love NLU!')

        df = fitted_pipe.predict('I love to go to the super market when there are great offers!')
        print(df.columns)
        print(df.pos)

    def load_classifier_dl_dataset(self):
        # catual url kagge
        train_url = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/fr/pos/UD_French/UD_French-GSD_2.3.txt '
        path = None



        return pd.DataFrame(path)
if __name__ == '__main__':
    unittest.main()

