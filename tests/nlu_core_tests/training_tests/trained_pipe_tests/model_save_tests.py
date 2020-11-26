
import unittest
from nlu import *
class PipelineSavingTests(unittest.TestCase):

    def test_pipeline_save(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #
        store_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tmp/models1'
        store_path = store_path + '_model'
        nlu.load('emotion').save(store_path,overwrite=True)



    def test_saving_component(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #
        store_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tmp/models'
        store_path = store_path + '_model'
        pipe = nlu.load('emotion')
        pipe.print_info()
        pipe.save(store_path, component='classifier_dl', overwrite=True)

        # nlu.load('emotion').save(store_path)
        #
        # nlu.load('emotion').save(store_path,overwrite=True)

    def test_saving_trained_model(self):
        # Just put in one of the many special 'trainable' references, to load
        # trainable components into the pipe
        # nlu.load('ner classifier_dl bert') will only give trainable classifier dl
        #
        store_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tmp/models'

        test_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_test.csv'
        train_df = pd.read_csv(test_path)
        train_df.columns = ['y','text']
        pipe = nlu.load('train.classifier',verbose=True,)
        fitted_pipe = pipe.fit(train_df)
        fitted_pipe.save(store_path, overwrite=True)

if __name__ == '__main__':
    unittest.main()

