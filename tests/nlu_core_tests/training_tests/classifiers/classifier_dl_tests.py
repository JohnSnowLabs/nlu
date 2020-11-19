

from sklearn.metrics import classification_report

import unittest
from nlu import *
class ClassifierDlTests(unittest.TestCase):

    def test_classifier_dl_training(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #

        test_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_test.csv'

        # train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_train.csv'
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(test_path)
        train_df.columns = ['y','x']
        test_df.columns = ['y','x']
        pipe = nlu.load('train.classifier',verbose=True,)
        pipe['classifier_dl'].setMaxEpochs(2)
        fitted_model = pipe.fit(train_df)
        df = fitted_model.predict(train_df)
        print(df[['category','label']])

        df = fitted_model.predict(test_df)
        print(df.columns)
        print(df[['category','label']])
        print (classification_report(df['label'], df['category']))


    def test_classifier_dl_training_stacked(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #

        test_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_test.csv'

        # train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_train.csv'
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(test_path)
        train_df.columns = ['label','text']
        test_df.columns = ['label','text']
        pipe = nlu.load('train.classifier emotion',verbose=True,)
        pipe['classifier_dl'].setMaxEpochs(2)
        df = pipe.predict(train_df,output_level='document')
        print(df.columns)
        print(df[['category']])
# Label column missing!
        df = pipe.predict(test_df,output_level='document')
        print(df.columns)
        print(df[['category']])

        from sklearn.metrics import classification_report

        print (classification_report(df['label'], df['category']))

    def test_classifier_dl_training_stacked2(self):

        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #
        # ONLY TRAIN 1 MODEL AT 1 TIME! No support multi model train
        # pipe = nlu.load('train.classifier train.pos',verbose=True,)

        # todo next NER
        test_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_test.csv'

        # train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_train.csv'
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(test_path)

        train_df.columns = ['label','text']
        test_df.columns = ['label','text']

        pipe = nlu.load('train.classifier emotion pos',verbose=True,)
        pipe['classifier_dl'].setMaxEpochs(2)

        #rather use pipe.fit()
        # trained_pipe = pipe.fit(data_with_label).save(path)
        # preds = trained_pipe.predict(data)
        #
        pipe.fit(train_df,output_level='document')

        df = pipe.fit_predict(train_df,output_level='document')


        print(df.columns)
        print(df[['category']])
        print (classification_report(df['label'], df['category']))

        # Label column missing!
        df = pipe.predict(test_df,output_level='document')
        print(df.columns)
        print(df[['category']])


        print (classification_report(df['label'], df['category']))

    def load_classifier_dl_dataset(self):
        # train_url = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_train.csv'
        # test_url = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv'
        path = None


        return pd.DataFrame(path)
if __name__ == '__main__':
    unittest.main()

