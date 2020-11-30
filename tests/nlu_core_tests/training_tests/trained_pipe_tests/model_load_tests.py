import tests.test_utils as t
import unittest
from nlu import *
class PipelineLoadingTests(unittest.TestCase):


    def test_pipeline_load_from_hdd(self):
        store_path = self.get_model_dir('trained_pipe_load')
        test_path = self.load_classifier_dl_dataset()
        train_df = pd.read_csv(test_path)
        train_df.columns = ['y','text']
        pipe = nlu.load('train.classifier',verbose=True,)
        fitted_pipe = pipe.fit(train_df)
        fitted_pipe.save(store_path, overwrite=True)
        loaded_pipe = nlu.load(path=store_path)
        print(loaded_pipe.predict('I Love offline mode!'))

    def load_classifier_dl_dataset(self):
        #relative from tests/nlu_core_tests/training_tests/trained_pipe_tests
        output_file_name = 'news_category_test.csv'
        output_folder = 'classifier_dl/'
        data_dir = '../../../datasets/'
        data_url = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_train.csv"
        return t.download_dataset(data_url,output_file_name,output_folder,data_dir)

    def get_model_dir(self,suffix):
        #relative from tests/nlu_core_tests/training_tests/trained_pipe_tests
        output_folder = f'classifier_dl_save{suffix}/'
        model_dir = '../../../models/'
        output_path = model_dir+output_folder
        t.create_path_if_not_exist(output_path)
        return output_folder




if __name__ == '__main__':
    unittest.main()

