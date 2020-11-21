

from sklearn.metrics import classification_report

import unittest
from nlu import *
class SentimentTrainingTests(unittest.TestCase):

    def test_sentiment_training(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('sentiment classifier_dl bert') will only give trainable classifier dl
        #


        #sentiment datase
        train_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/sentiment_dl/aclimdb_train.csv'
        # test_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/sentiment_dl/AllProductReviews.csv'
        test_path='/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/sentiment_dl/aclimdb_train.csv'
        df_train = pd.read_csv(test_path,error_bad_lines=False)
        #convert int to str labels so our model predicts strings not numbers
        label_mappings = {
            1 : '1 star',
            2 : '2 star',
            3 : '3 star',
            4 : '4 star',
            5 : '5 star',
        }
        df_train.columns = ['text','y']
        # df_train['y'] = df_train.ReviewStar
        # df_train.replace({y : label_mappings})
        # df_train['text'] = df_train.ReviewBody

        df_train.dropna(inplace=True)
        pipe = nlu.load('train.sentiment',verbose=True)
        fitted_pipe = pipe.fit(df_train.iloc[0:50])
        # df = fitted_pipe.predict(' I love NLU!')

        df = fitted_pipe.predict(df_train.iloc[0:50])
        print(df)
        print(df.columns)


    def load_classifier_dl_dataset(self):
        # catual url kagge
        # https://www.kaggle.com/shitalkat/amazonearphonesreviews
        # train_url = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sentiment-corpus/aclimdb/aclimdb_train.csv'
        # test_url =' https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sentiment-corpus/aclimdb/aclimdb_test.csv'
        path = None



        return pd.DataFrame(path)
if __name__ == '__main__':
    unittest.main()

