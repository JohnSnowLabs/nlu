import unittest

import pandas as pd

import tests.test_utils as t
from nlu import *


class PipelineLoadingTests(unittest.TestCase):
    def test_pipeline_load_from_hdd_after_training(self):
        train_df = self.load_classifier_dl_dataset()
        train_df.columns = ["y", "text"]
        pipe = nlu.load(
            "train.classifier",
            verbose=True,
        )
        pipe = pipe.fit(train_df)
        store_path = t.create_model_dir_if_not_exist_and_get_path()
        pipe.save(store_path, overwrite=True)
        print(pipe.predict("I Love offline mode!"))
        # Too heavy for Github actions  :
        # component_list = nlu.load(path=store_path)
        # print(component_list.predict('I Love offline mode!'))

    def test_pipeline_load_from_hdd_from_spark_nlp(self):
        p_path = "/home/ckl/Downloads/tmp/analyze_sentiment_en_3.0.0_3.0_1616544471011"
        p = nlu.load(path=p_path)
        res = p.predict("I love offline mode")
        for c in res:
            print(res[c])

    def test_model_load_from_hdd_from_spark_nlp(self):
        m_path = "/home/ckl/Downloads/tmp/pos_afribooms_af_3.0.0_3.0_1617749039095"
        p = nlu.load(path=m_path)
        res = p.predict("I love offline mode")
        for c in res:
            print(res[c])

    def load_classifier_dl_dataset(self):
        output_file_name = "news_category_test.csv"
        output_folder = "classifier_dl/"
        data_url = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv"

        return pd.read_csv(
            t.download_dataset(data_url, output_file_name, output_folder)
        ).iloc[0:100]


if __name__ == "__main__":
    unittest.main()
