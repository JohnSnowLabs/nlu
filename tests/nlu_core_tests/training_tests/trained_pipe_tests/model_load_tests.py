
import unittest
from nlu import *
class PipelineLoadingTests(unittest.TestCase):

    def test_pipeline_load(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #
        store_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tmp/models1'
        store_path = store_path + '_model'
        nlu.load('emotion').save(store_path,overwrite=True)
        loaded_pipe = nlu.load(path=store_path)
        loaded_pipe.predict('I Love offline mode!')

    def test_loading_model(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #
        store_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tmp/models'
        # store_path = store_path + '_model'
        # pipe = nlu.load('sentiment')
        # pipe.predict('You gotta predict to make a pipe fitted')
        # pipe.print_info()
        # pipe.save(store_path, overwrite=True)

        # loaded_pipe = PipelineModel.load(path=store_path)
        pipe = nlu.load(path=store_path,verbose=True)
        df = pipe.predict("I love loading models from hdd")
        print('huzza!')
        print(df)
        print(df.columns)

        # nlu.load('emotion').save(store_path)
        #
        # nlu.load('emotion').save(store_path,overwrite=True)

    def test_loading_trained_model(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #
        store_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tmp/models'

        test_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_test.csv'
        train_df = pd.read_csv(test_path)
        train_df.columns = ['label','text']
        pipe = nlu.load('train.classifier emotion',verbose=True,)
        pipe = pipe.fit(train_df)
        pipe.save(store_path, overwrite=True)
        loaded_pipe = nlu.load('emotion', store_path)
        loaded_pipe.predict('I Love offline mode!')

if __name__ == '__main__':
    unittest.main()

