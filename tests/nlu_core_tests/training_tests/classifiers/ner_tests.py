

from sklearn.metrics import classification_report

import unittest
from nlu import *
import tests.test_utils as t
class NerTrainingTests(unittest.TestCase):

    def test_ner_training(self):
        #CONLL data
        train_path = self.load_ner_train_dataset_and_get_path()
        pipe = nlu.load('train.ner',verbose=True)
        fitted_pipe = pipe.fit(dataset_path=train_path)
        df = fitted_pipe.predict(' Hello Donald Trump and Hello Angela Merkel')
        print(df)
        print(df.columns)
    # Too heavy running on github actions

    # def test_ner_training_with_custom_embeddings(self):
    #     # Test using custom pipe embed composistion
    #     #CONLL data
    #     # train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/ner/eng.train.txt'
    #     train_path = self.load_ner_train_dataset_and_get_path()
    #
    #     pipe = nlu.load('elmo train.ner',verbose=True)
    #     fitted_pipe = pipe.fit(dataset_path=train_path)
    #     df = fitted_pipe.predict(' Hello Donald Trump and Hello Angela Merkel')
    #     print(df)
    #     print(df.columns)
    #
    #
    # def test_ner_training_with_custom_embeddings_and_pos(self):
    #     # Use italian POS and ELMO embeds to train the NER model!
    #     #CONLL data
    #     # train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/ner/eng.train.txt'
    #     train_path = self.load_ner_train_dataset_and_get_path()
    #
    #     pipe = nlu.load('elmo it.pos train.ner',verbose=True)
    #     fitted_pipe = pipe.fit(dataset_path=train_path)
    #     df = fitted_pipe.predict(' Hello Donald Trump and Hello Angela Merkel')
    #     print(df)
    #     print(df.columns)
    #





    def load_ner_train_dataset_and_get_path(self):
        output_file_name = 'conll2008.data'
        output_folder = 'ner/'
        data_url = "https://github.com/patverga/torch-ner-nlp-from-scratch/raw/master/data/conll2003/eng.train"
        return t.download_dataset(data_url,output_file_name,output_folder)




if __name__ == '__main__':
    unittest.main()

