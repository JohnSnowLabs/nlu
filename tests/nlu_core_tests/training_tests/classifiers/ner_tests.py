

from sklearn.metrics import classification_report

import unittest
from nlu import *

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



    def test_load_ner_train_dataset(self,data_url,output_file_name,output_folder,data_dir,):
        import urllib.request
        import os
        download_path = data_dir + output_folder + output_file_name

        #Check if dir exists, if not create it
        if not os.path.exists(data_dir + output_folder):
            print('Creating dir',data_dir + output_folder)
            os.mkdir(data_dir + output_folder)

        from pathlib import Path
        #Check if file exists, if not download it
        if not Path(download_path).is_file():
            urllib.request.urlretrieve(data_url, download_path )

        print('Downloaded dataset to ',download_path)
        return download_path

    def load_ner_train_dataset_and_get_path(self):
        #relative from tests/nlu_core_tests/training_tests/classifiers
        output_file_name = 'ner_eng.train'
        output_folder = 'ner/'
        data_dir = '../../../datasets/'
        conll_data_url = "https://github.com/patverga/torch-ner-nlp-from-scratch/raw/master/data/conll2003/eng.train"
        return self.test_load_ner_train_dataset(conll_data_url,output_file_name,output_folder,data_dir)





if __name__ == '__main__':
    unittest.main()

