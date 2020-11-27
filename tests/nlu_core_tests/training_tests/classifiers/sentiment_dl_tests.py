

from sklearn.metrics import classification_report
import tests.test_utils as t
import unittest
from nlu import *
class SentimentTrainingTests(unittest.TestCase):

    def test_sentiment_training(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('sentiment classifier_dl bert') will only give trainable classifier dl
        #


        #sentiment datase
        test_path = self.load_sentiment_dl_dataset()#'/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/sentiment_dl/AllProductReviews.csv'
        df_train = pd.read_csv(test_path,error_bad_lines=False)
        #convert int to str labels so our model predicts strings not numbers
        # the text data to use for classification should be in a column named 'text'
        df_train['text'] = df_train['ReviewTitle']
        # the label column must have name 'y' name be of type str
        df_train['y'] = df_train['ReviewStar']#.astype(str)
        pipe = nlu.load('train.sentiment',verbose=True)
        fitted_pipe = pipe.fit(df_train.iloc[0:50])
        # df = fitted_pipe.predict(' I love NLU!')

        df = fitted_pipe.predict(df_train.iloc[0:50])
        print(df)
        print(df.columns)


    def load_sentiment_dl_dataset(self):
        #relative from tests/nlu_core_tests/training_tests/classifiers
        # https://www.kaggle.com/shitalkat/amazonearphonesreviews
        output_file_name = 'amazon_sentiment_reviews.csv'
        output_folder = 'classifier_dl/'
        data_dir = '../../../datasets/'
        data_url = 'http://ckl-it.de/wp-content/uploads/2020/11/AllProductReviews-1.csv'
        return t.download_dataset(data_url,output_file_name,output_folder,data_dir)




        return pd.DataFrame(path)
if __name__ == '__main__':
    unittest.main()

