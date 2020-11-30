

from sklearn.metrics import classification_report

import unittest
from nlu import *
import tests.test_utils as t


class ClassifierDlTests(unittest.TestCase):

    def test_classifier_dl_training(self):
        test_path = self.load_classifier_dl_dataset()
        test_df = pd.read_csv(test_path)
        train_df = test_df
        train_df.columns = ['y','text']
        test_df.columns = ['y','text']
        pipe = nlu.load('train.classifier',verbose=True,)
        pipe['classifier_dl'].setMaxEpochs(2)
        fitted_model = pipe.fit(train_df)
        df = fitted_model.predict(train_df)
        print(df[['category','y']])

        df = fitted_model.predict(test_df)
        print(df.columns)
        print(df[['category','y']])
        print (classification_report(df['y'], df['category']))


    def test_classifier_dl_custom_embeds_doc_level(self):
        test_path = self.load_classifier_dl_dataset()
        test_df = pd.read_csv(test_path)
        train_df = test_df
        train_df.columns = ['y','text']
        test_df.columns = ['y','text']
        pipe = nlu.load('embed_sentence.bert train.classifier',verbose=True,)
        pipe['classifier_dl'].setMaxEpochs(2)
        fitted_model = pipe.fit(train_df)
        df = fitted_model.predict(train_df, output_level='document')
        print(df.columns)
        print(df[['category','y']])
        df = fitted_model.predict(test_df, output_level='document')
        print(df.columns)
        print(df[['category','y']])

        # Eval results
        from sklearn.metrics import classification_report

        print (classification_report(df['y'], df['category']))

    def test_classifier_dl_custom_embeds_sentence_level(self):
        test_path = self.load_classifier_dl_dataset()
        test_df = pd.read_csv(test_path)
        train_df = test_df
        train_df.columns = ['y','text']
        test_df.columns = ['y','text']
        pipe = nlu.load('embed_sentence.bert train.classifier',verbose=True,)
        pipe['classifier_dl'].setMaxEpochs(2)
        fitted_model = pipe.fit(train_df)
        df = fitted_model.predict(train_df, output_level='sentence')

        print(df.columns)
        print(df[['category','y']])
        df = fitted_model.predict(test_df, output_level='sentence')
        print(df.columns)
        print(df[['category','y']])

        # Eval results
        from sklearn.metrics import classification_report

        print (classification_report(df['y'], df['category']))


    def test_classifier_dl_custom_embeds_auto_level(self):
        test_path = self.load_classifier_dl_dataset()
        test_df = pd.read_csv(test_path)
        train_df = test_df
        train_df.columns = ['y','text']
        test_df.columns = ['y','text']
        pipe = nlu.load('embed_sentence.bert train.classifier',verbose=True,)
        pipe['classifier_dl'].setMaxEpochs(2)
        fitted_model = pipe.fit(train_df)
        df = fitted_model.predict(train_df)
        print(df.columns)
        print(df[['category','y']])
        df = fitted_model.predict(test_df)
        print(df.columns)
        print(df[['category','y']])

        # Eval results
        from sklearn.metrics import classification_report

        print (classification_report(df['y'], df['category']))
    def load_classifier_dl_dataset(self):
        #relative from tests/nlu_core_tests/training_tests/classifiers
        output_file_name = 'news_category_test.csv'
        output_folder = 'classifier_dl/'
        data_dir = '../../../datasets/'
        data_url = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv"
        return t.download_dataset(data_url,output_file_name,output_folder,data_dir)
    #
    # def load_classifier_dl_dataset(self):
    #     #relative from tests/nlu_core_tests/training_tests/classifiers
    #     output_file_name = 'news_category_test.csv'
    #     output_folder = 'classifier_dl/'
    #     data_dir = t.create_dataset_dir_if_not_exist_and_get_path()
    #     t.create_path_if_not_exist(data_dir + output_file_name)
    #     data_url = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv"
    #     return t.download_dataset(data_url,output_file_name,output_folder,data_dir)



if __name__ == '__main__':
    unittest.main()

