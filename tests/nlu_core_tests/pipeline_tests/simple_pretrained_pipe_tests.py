import unittest

from nlu import *


class PretrainedPipeTests(unittest.TestCase):
    def simple_pretrained_pipe_tests(self):
        df = nlu.load("ner.onto", verbose=True).predict("I love peanutbutter and jelly")
        for c in df.columns:
            print(df[c])

    # def test_offline_load_pipe(self):
    #     pipe_path ='/home/ckl/cache_pretrained/analyze_sentimentdl_use_imdb_en_2.7.1_2.4_1610723836151'
    #     df = nlu.load(path = pipe_path,verbose=True).predict('I love peanutbutter and jelly')
    #     for c in df.columns: print(df[c])
    # def test_offline_load_model(self):
    #     model_path ='/home/ckl/cache_pretrained/stopwords_hi_hi_2.5.4_2.4_1594742439035'
    #     model_path = '/home/ckl/cache_pretrained/bert_token_classifier_ner_ud_gsd_ja_3.2.2_3.0_1631279615344'
    #     df = nlu.load(path = model_path,verbose=True).predict('I love peanutbutter and jelly')
    #     for c in df.columns: print(df[c])


if __name__ == "__main__":
    unittest.main()
