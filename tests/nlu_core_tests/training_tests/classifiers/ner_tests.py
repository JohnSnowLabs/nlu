

from sklearn.metrics import classification_report

import unittest
from nlu import *
import tests.test_utils as t
class NerTrainingTests(unittest.TestCase):
    def test_ner_training(self):
        # CONLL data
        train_path = self.load_ner_train_dataset_and_get_path()
        pipe = nlu.load('train.ner', verbose=True)
        pipe = pipe.fit(dataset_path=train_path)
        df = pipe.predict(' Hello Donald Trump and Hello Angela Merkel')
        pipe.save('saved_test_models/ner_training')
        for c in df.columns: print(df[c])


    def test_sentiment_with_embed_save(self):
        df_train = self.load_ner_train_dataset_and_get_path()
        # the text data to use for classification should be in a column named 'text'
        df_train['text'] = df_train['text_data']
        # the label column must have name 'y' name be of type str

        df_train['Sentiment'] = df_train['Sentiment']
        df_train['y'] = df_train['Sentiment'].astype(str)
        df_train.y = df_train.y.str.replace('-1','negative')
        df_train.y = df_train.y.str.replace('1','positive')
        # df_train=df_train.iloc[0:4000]

        pipe = nlu.load('en.embed_sentence.electra train.sentiment',verbose=True, )
        pipe.print_info()
        pipe['sentiment_dl'].setMaxEpochs(6)
        pipe = pipe.fit(df_train)
        # df = fitted_pipe.predict(' I love NLU!')

        df = pipe.predict(df_train.iloc[0:50],output_level='sentence')
        print(df)
        for c in df.columns : print (df[c])

# Too heavy running on github actions

    # def test_ner_training_with_custom_embeddings(self):
    #     # Test using custom component_list embed composistion
    #     #CONLL data
    #     # train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/ner/eng.train.txt'
    #     train_path = self.load_ner_train_dataset_and_get_path()
    #
    #     component_list = nlu.load('elmo train.ner',verbose=True)
    #     fitted_pipe = component_list.fit(dataset_path=train_path)
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
    #     component_list = nlu.load('elmo it.pos train.ner',verbose=True)
    #     fitted_pipe = component_list.fit(dataset_path=train_path)
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

