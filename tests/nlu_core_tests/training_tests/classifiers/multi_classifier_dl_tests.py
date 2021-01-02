

from sklearn.metrics import classification_report

import unittest
from nlu import *
import tests.test_utils as t


class MultiClassifierDlTests(unittest.TestCase):

    def test_multi_classifier_dl_training(self):
        # The y column must be a string seperated with ```,``` . Custom seperators can be configured by passing
        test_df = self.load_multi_classifier_dl_dataset()
        # test_df.columns = ['y_str','text']
        test_df.columns = ['y','text']

        # test_df['y'] = test_df.y_str.str.split(',')

        # test_df.y = test_df.y.astype('stringArray')#pd.arrays.
        # test_df.y = test_df.y.astype(list[str])#pd.arrays.

        print(test_df.y)
        print(test_df)
        print(test_df.dtypes)

        # test_df.drop('y_str',inplace=True,axis=1)
        train_df = test_df

        pipe = nlu.load('train.multi_classifier',verbose=True,)
#: java.lang.IllegalArgumentException: requirement failed: The label column MultiClassifierDLApproach_cbfe97978b3c__labelColumn type is StringType and it's not compatible. Compatible types are ArrayType(StringType).

        # pipe['multi_classifier_dl'].setMaxEpochs(2)
        pipe.print_info()
        pipe = pipe.fit(train_df)
        df = pipe.predict(train_df)
        print(df.columns)
        print(df[['multi_classifier_classes','y']])
        print(df[['multi_classifier_confidences','y']])

        df = pipe.predict(test_df)
        print(df.columns)
        print(df[['multi_classifier_classes','y']])
        print(df[['multi_classifier_confidence','y']])
        df.dropna(inplace=True)
        print (classification_report(df['y'], df['multi_classifier_classes']))
    # Too heavy running on github actions

    #
    # def test_multi_classifier_dl_custom_embeds_doc_level(self):
    #     test_df = self.load_multi_classifier_dl_dataset()
    #     # test_df.columns = ['y_str','text']
    #     test_df.columns = ['y','text']
    #
    #
    #
    #     print(test_df.y)
    #     print(test_df)
    #     print(test_df.dtypes)
    #
    #     # test_df.drop('y_str',inplace=True,axis=1)
    #     train_df = test_df
    #
    #     pipe = nlu.load('embed_sentence.bert    train.multi_classifier',verbose=True,)
    #     #: java.lang.IllegalArgumentException: requirement failed: The label column MultiClassifierDLApproach_cbfe97978b3c__labelColumn type is StringType and it's not compatible. Compatible types are ArrayType(StringType).
    #
    #     # pipe['multi_classifier_dl'].setMaxEpochs(2)
    #     pipe.print_info()
    #     pipe = pipe.fit(train_df)
    #     df = pipe.predict(train_df)
    #     print(df.columns)
    #     print(df[['multi_classifier','y']])
    #     print(df[['multi_classifier_confidence','y']])
    #     df = pipe.predict(test_df)
    #     print(df.columns)
    #     print(df[['multi_classifier','y']])
    #     print(df[['multi_classifier_confidence','y']])
    #     df.dropna(inplace=True)
    #     print (classification_report(df['y'], df['multi_classifier']))
    #
    # def test_multi_classifier_dl_custom_embeds_sentence_level(self):
    #     test_path = self.load_multi_classifier_dl_dataset()
    #     test_df = pd.read_csv(test_path)
    #     train_df = test_df
    #     train_df.columns = ['y','text']
    #     test_df.columns = ['y','text']
    #     pipe = nlu.load('embed_sentence.bert train.multi_classifier',verbose=True,)
    #     pipe['multi_classifier_dl'].setMaxEpochs(2)
    #     pipe = pipe.fit(train_df)
    #     df = pipe.predict(train_df, output_level='sentence')
    #     print(df.columns)
    #     print(df[['category','y']])
    #     df = pipe.predict(test_df, output_level='sentence')
    #     print(df.columns)
    #     print(df[['category','y']])
    #     # Eval results
    #     from sklearn.metrics import classification_report
    #     print (classification_report(df['y'], df['category']))
    #
    #
    # def test_multi_classifier_dl_custom_embeds_auto_level(self):
    #     test_path = self.load_multi_classifier_dl_dataset()
    #     test_df = pd.read_csv(test_path)
    #     train_df = test_df
    #     train_df.columns = ['y','text']
    #     test_df.columns = ['y','text']
    #     pipe = nlu.load('embed_sentence.bert train.multi_classifier',verbose=True,)
    #     pipe['multi_classifier_dl'].setMaxEpochs(2)
    #     pipe = pipe.fit(train_df)
    #     df = pipe.predict(train_df)
    #     print(df.columns)
    #     print(df[['category','y']])
    #     df = pipe.predict(test_df)
    #     print(df.columns)
    #     print(df[['category','y']])
    #     # Eval results
    #     from sklearn.metrics import classification_report
    #     print (classification_report(df['y'], df['category']))


    # def load_multi_classifier_dl_dataset(self):
    #     #relative from tests/nlu_core_tests/training_tests/classifiers
    #     p = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/multi_classifier_dl/e2e-dataset/testset_w_refs.csv'
    #     return pd.read_csv(p)

    def load_multi_classifier_dl_dataset(self):
        output_file_name = 'e2e_test.csv'
        output_folder = 'multi_classifier_dl/'
        # data_url = "http://ckl-it.de/wp-content/uploads/2020/12/testset_w_refs.csv"
        data_url = "http://ckl-it.de/wp-content/uploads/2020/12/e2e.csv"

        return pd.read_csv(t.download_dataset(data_url,output_file_name,output_folder)).iloc[0:100]

        # output_file_name = 'news_category_test.csv'
        # output_folder = 'multi_classifier_dl/'
        # data_dir = '../../../datasets/'
        # data_url = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv"
        # return t.download_dataset(data_url,output_file_name,output_folder,data_dir)
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

