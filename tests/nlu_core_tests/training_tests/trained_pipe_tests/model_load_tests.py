import tests.test_utils as t
import unittest
from nlu import *
class PipelineLoadingTests(unittest.TestCase):


    def test_pipeline_load_from_hdd(self):
        train_df = self.load_classifier_dl_dataset()
        train_df.columns = ['y','text']
        pipe = nlu.load('train.classifier',verbose=True,)
        fitted_pipe = pipe.fit(train_df)
        store_path = t.create_model_dir_if_not_exist_and_get_path()
        fitted_pipe.save(store_path, overwrite=True)
        loaded_pipe = nlu.load(path=store_path)
        print(loaded_pipe.predict('I Love offline mode!'))

    def load_classifier_dl_dataset(self):
        #relative from tests/nlu_core_tests/training_tests/trained_pipe_tests
        output_file_name = 'news_category_test.csv'
        output_folder = 'classifier_dl/'
        data_url = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv"

        return pd.read_csv(t.download_dataset(data_url,output_file_name,output_folder)).iloc[0:100]




if __name__ == '__main__':
    unittest.main()

