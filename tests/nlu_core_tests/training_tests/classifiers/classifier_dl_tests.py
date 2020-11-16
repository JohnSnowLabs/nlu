


import unittest
from nlu import *
class ClassifierDlTests(unittest.TestCase):

    def test_classifier_dl_training(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #

        test_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_test.csv'

        train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_train.csv'
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(train_path)
        train_df.columns = ['label','text']
        pipe = nlu.load('train.classifier pos ner',verbose=True,)
        pipe['classifier_dl'].setMaxEpochs(2)
        df = pipe.predict(train_df)
        print(df[['category','label']])

        pipe = nlu.load('train.classifier pos ner',verbose=True,)
        df = pipe.predict(test_df)
        print(df[['category','label']])

        from sklearn.metrics import classification_report

        print (classification_report(df['label'], df['category']))


    def load_classifier_dl_dataset(self):
        # train_url = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_train.csv'
        # test_url = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv'
        path = None


        return pd.DataFrame(path)
if __name__ == '__main__':
    unittest.main()

