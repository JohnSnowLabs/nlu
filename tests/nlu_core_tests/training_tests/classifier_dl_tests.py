import unittest

import pandas as pd

import tests.test_utils as t
from nlu import *


class ClassifierDlTests(unittest.TestCase):
    def test_classifier_dl_training_labse(self):
        test_df = self.load_classifier_dl_dataset()
        train_df = test_df
        train_df.columns = ["y", "text"]
        test_df.columns = ["y", "text"]
        pipe = nlu.load(
            "xx.embed_sentence.labse train.classifier",
            verbose=True,
        )
        pipe["classifier_dl"].setMaxEpochs(2)
        pipe = pipe.fit(train_df)
        df = pipe.predict(test_df)
        pipe.print_info()
        print(df.columns)
        for c in df.columns:
            print(df[c])

    def test_classifier_dl_training(self):
        test_df = self.load_classifier_dl_dataset()
        train_df = test_df
        train_df.columns = ["y", "text"]
        test_df.columns = ["y", "text"]
        pipe = nlu.load(
            "train.classifier",
            verbose=True,
        )
        # pipe['classifier_dl'].setMaxEpochs(2)
        pipe = pipe.fit(train_df)
        df = pipe.predict(train_df)
        # print(df[['category','y']])

        df = pipe.predict(test_df)
        pipe.print_info()
        print(df.columns)
        # print(df[['category','y']])
        # print (classification_report(df['y'], df['category']))
        for c in df.columns:
            print(df[c])
        # component_list.save('/home/loan/Documents/freelance/jsl/nlu/nlu4realgit/tests/trained_models/quick_classifi')

    # Too heavy running on github actions
    # def test_classifier_dl_custom_embeds_doc_level(self):
    #     test_df = self.load_classifier_dl_dataset()
    #     train_df = test_df
    #     train_df.columns = ['y','text']
    #     test_df.columns = ['y','text']
    #     component_list = nlu.load('embed_sentence.bert train.classifier',verbose=True,)
    #     component_list['classifier_dl'].setMaxEpochs(2)
    #     fitted_model = component_list.fit(train_df)
    #     df = fitted_model.predict(train_df, output_level='document')
    #     print(df.columns)
    #     print(df[['category','y']])
    #     df = fitted_model.predict(test_df, output_level='document')
    #     print(df.columns)
    #     print(df[['category','y']])
    #
    #     # Eval results
    #     from sklearn.metrics import classification_report
    #
    #     print (classification_report(df['y'], df['category']))
    #
    # def test_classifier_dl_custom_embeds_sentence_level(self):
    #     test_df = self.load_classifier_dl_dataset()
    #     train_df = test_df
    #     train_df.columns = ['y','text']
    #     test_df.columns = ['y','text']
    #     component_list = nlu.load('embed_sentence.bert train.classifier',verbose=True,)
    #     component_list['classifier_dl'].setMaxEpochs(2)
    #     fitted_model = component_list.fit(train_df)
    #     df = fitted_model.predict(train_df, output_level='sentence')
    #
    #     print(df.columns)
    #     print(df[['category','y']])
    #     df = fitted_model.predict(test_df, output_level='sentence')
    #     print(df.columns)
    #     print(df[['category','y']])
    #
    #     # Eval results
    #     from sklearn.metrics import classification_report
    #
    #     print (classification_report(df['y'], df['category']))
    #
    #
    # def test_classifier_dl_custom_embeds_auto_level(self):
    #     test_df = self.load_classifier_dl_dataset()
    #     train_df = test_df
    #     train_df.columns = ['y','text']
    #     test_df.columns = ['y','text']
    #     component_list = nlu.load('embed_sentence.bert train.classifier',verbose=True,)
    #     component_list['classifier_dl'].setMaxEpochs(2)
    #     fitted_model = component_list.fit(train_df)
    #     df = fitted_model.predict(train_df)
    #     print(df.columns)
    #     print(df[['category','y']])
    #     df = fitted_model.predict(test_df)
    #     print(df.columns)
    #     print(df[['category','y']])
    #
    #     # Eval results
    #     from sklearn.metrics import classification_report
    #
    #     print (classification_report(df['y'], df['category']))

    def load_classifier_dl_dataset(self):
        output_file_name = "news_category_test.csv"
        output_folder = "classifier_dl/"
        data_url = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv"
        return pd.read_csv(
            t.download_dataset(data_url, output_file_name, output_folder)
        ).iloc[0:15]


if __name__ == "__main__":
    unittest.main()
