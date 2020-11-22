

from sklearn.metrics import classification_report

import unittest
from nlu import *
class NerTrainingTests(unittest.TestCase):

    def test_ner_training(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #

        # test_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_test.csv'

        #CONLL data
        train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/ner/eng.train.txt'

        pipe = nlu.load('train.ner',verbose=True)
        fitted_pipe = pipe.fit(dataset_path=train_path)
        df = fitted_pipe.predict(' Hello Donald Trump and Hello Angela Merkel')
        print(df)
        print(df.columns)


    def load_classifier_dl_dataset(self):
        # train_url = "https://github.com/patverga/torch-ner-nlp-from-scratch/raw/master/data/conll2003/eng.train"
        # test_url = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv'
        path = None


        return pd.DataFrame(path)
if __name__ == '__main__':
    unittest.main()

