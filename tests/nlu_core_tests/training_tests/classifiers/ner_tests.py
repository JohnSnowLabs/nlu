

from sklearn.metrics import classification_report

import unittest
from nlu import *
import tests.test_utils as t
class NerTrainingTests(unittest.TestCase):

    def test_ner_training(self):
        #CONLL data
        # train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/ner/eng.train.txt'
        train_path = self.load_ner_train_dataset_and_get_path()
        pipe = nlu.load('train.ner',verbose=True)
        fitted_pipe = pipe.fit(dataset_path=train_path)
        df = fitted_pipe.predict(' Hello Donald Trump and Hello Angela Merkel')
        print(df)
        print(df.columns)

    def test_ner_training_with_custom_embeddings(self):
        # Test using custom pipe embed composistion
        #CONLL data
        # train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/ner/eng.train.txt'
        train_path = self.load_ner_train_dataset_and_get_path()

        pipe = nlu.load('elmo train.ner',verbose=True)
        fitted_pipe = pipe.fit(dataset_path=train_path)
        df = fitted_pipe.predict(' Hello Donald Trump and Hello Angela Merkel')
        print(df)
        print(df.columns)


    def test_ner_training_with_custom_embeddings_and_pos(self):
        # Use italian POS and ELMO embeds to train the NER model!
        #CONLL data
        # train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/ner/eng.train.txt'
        train_path = self.load_ner_train_dataset_and_get_path()

        pipe = nlu.load('elmo it.pos train.ner',verbose=True)
        fitted_pipe = pipe.fit(dataset_path=train_path)
        df = fitted_pipe.predict(' Hello Donald Trump and Hello Angela Merkel')
        print(df)
        print(df.columns)





    def load_ner_train_dataset_and_get_path(self):
        #relative from tests/nlu_core_tests/training_tests/classifiers
        output_file_name = 'ner_eng.train'
        output_folder = 'ner/'
        data_dir = '../../../datasets/'
        data_dir = t.create_dataset_dir_if_not_exist_and_get_path()
        t.create_path_if_not_exist(data_dir + output_file_name)
        conll_data_url = "https://github.com/patverga/torch-ner-nlp-from-scratch/raw/master/data/conll2003/eng.train"
        return t.download_dataset(conll_data_url,output_file_name,output_folder,data_dir)





if __name__ == '__main__':
    unittest.main()

