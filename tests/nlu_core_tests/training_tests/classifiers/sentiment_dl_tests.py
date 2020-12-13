

from sklearn.metrics import classification_report
import tests.test_utils as t
import unittest
from nlu import *
class SentimentTrainingTests(unittest.TestCase):

    def test_sentiment_training(self):

        #sentiment datase
        df_train = self.load_sentiment_dl_dataset()#'/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/sentiment_dl/AllProductReviews.csv'
        print(df_train.columns)

        #convert int to str labels so our model predicts strings not numbers
        # the text data to use for classification should be in a column named 'text'
        df_train['text'] = df_train['text_data']
        # the label column must have name 'y' name be of type str
        df_train['y'] = df_train['Sentiment'].astype(str)
        df_train.y = df_train.y.astype(str)
        df_train.y = df_train.y.str.replace('-1','negative')
        df_train.y = df_train.y.str.replace('1','positive')
        df_train=df_train.iloc[0:4000]
        pipe = nlu.load('train.sentiment',verbose=True)
        pipe = pipe.fit(df_train)


        df = pipe.predict(df_train)
        print(df)
        print(df.columns)
        print(df)
        print(df.columns)
        print(df[['sentiment','sentiment_confidence']])
        print(df.sentiment.value_counts())
        print(df.sentiment_confidence.value_counts())








    def test_sentiment_training_with_custom_embeds_document_level(self):

        #sentiment datase
        df_train = self.load_sentiment_dl_dataset()
        # the text data to use for classification should be in a column named 'text'
        df_train['text'] = df_train['text_data']
        # the label column must have name 'y' name be of type str

        df_train['Sentiment'] = df_train['Sentiment']
        df_train['y'] = df_train['Sentiment'].astype(str)
        df_train.y = df_train.y.str.replace('-1','negative')
        df_train.y = df_train.y.str.replace('1','positive')
        # df_train=df_train.iloc[0:4000]
        pipe = nlu.load('use train.sentiment',verbose=True, )
        pipe = pipe.fit(df_train)
        # df = fitted_pipe.predict(' I love NLU!')

        df = pipe.predict(df_train.iloc[0:500],output_level='document')
        print(df)
        print(df.columns)
        print(df[['sentiment','sentiment_confidence']])
        print(df.sentiment.value_counts())
        print(df.sentiment_confidence.value_counts())
# TODO test if bad performance persists in Spark NLP with non USE sentence eebddigns
    def test_sentiment_training_with_custom_embeds_sentence_level(self):

        #sentiment datase
        df_train = self.load_sentiment_dl_dataset()
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
        print(df.columns)
        print(df)
        print(df.columns)
        print(df[['sentiment','sentiment_confidence']])
        print(df.sentiment.value_counts())
        print(df.sentiment_confidence.value_counts())


    def load_sentiment_dl_dataset(self):
        output_file_name = 'e2e_test.csv'
        output_folder = 'multi_classifier_dl/'
        data_url = 'http://ckl-it.de/wp-content/uploads/2020/11/AllProductReviews-1.csv'
        return pd.read_csv(t.download_dataset(data_url,output_file_name,output_folder),error_bad_lines=False).iloc[0:100]




if __name__ == '__main__':
    unittest.main()

