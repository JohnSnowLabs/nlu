

from sklearn.metrics import classification_report
import tests.test_utils as t

import unittest
from nlu import *
class posTrainingTests(unittest.TestCase):

    def test_pos_training(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('pos classifier_dl bert') will only give trainable classifier dl
        #


        #pos datase
        train_path = self.load_pos_train_dataset_and_get_path()
        # df_train = pd.read_csv(train_path,error_bad_lines=False)
        #convert int to str labels so our model predicts strings not numbers
        # df_train.dropna(inplace=True)

        pipe = nlu.load('train.pos',verbose=True)
        pipe = pipe.fit(dataset_path=train_path, label_seperator='_')

        df = pipe.predict('I love to go to the super market when there are great offers!')
        print(df.columns)
        print(df.pos)


    def load_pos_train_dataset_and_get_path(self):
        output_file_name = 'ud_french.txt'
        output_folder = 'pos/'
        data_url = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/fr/pos/UD_French/UD_French-GSD_2.3.txt '
        return t.download_dataset(data_url,output_file_name,output_folder)


        return pd.DataFrame(path)
if __name__ == '__main__':
    unittest.main()

