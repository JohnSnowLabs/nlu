import unittest

import tests.test_utils as t
from nlu import *


class PipelineSavingTests(unittest.TestCase):
    def test_pipeline_save(self):
        store_path = t.create_model_dir_if_not_exist_and_get_path()
        nlu.load("emotion").save(store_path, overwrite=True)

    # def test_saving_component(self):
    #     # store_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tmp/models'
    #     store_path = t.create_model_dir_if_not_exist_and_get_path()
    #     component_list = nlu.load('emotion')
    #     component_list.print_info()
    #     component_list.save(store_path, component='classifier_dl', overwrite=True)

    def test_saving_trained_model(self):

        store_path = t.create_model_dir_if_not_exist_and_get_path()
        train_df = self.load_classifier_dl_dataset().iloc[0:100]

        # test_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets/news_category_test.csv'
        # store_path = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tmp/models'
        train_df.columns = ["y", "text"]
        pipe = nlu.load(
            "train.classifier",
            verbose=True,
        )
        fitted_pipe = pipe.fit(train_df)
        fitted_pipe.save(store_path, overwrite=True)

    def load_classifier_dl_dataset(self):
        output_file_name = "news_category_test.csv"
        output_folder = "classifier_dl/"
        data_url = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv"

        return pd.read_csv(
            t.download_dataset(data_url, output_file_name, output_folder)
        ).iloc[0:100]


if __name__ == "__main__":
    unittest.main()
