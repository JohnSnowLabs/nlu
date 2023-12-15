import unittest

import pandas as pd
from memory_profiler import memory_usage

import nlu
import tests.test_utils as t


class PandasTests(unittest.TestCase):
    def test_memory_batching_benchmark(self):
        test = PandasTests().test_data_batching
        ms = []
        for i in range(10):
            m = memory_usage((test))
            ms.append(sum(m))
            d = pd.DataFrame(m)
        d = pd.DataFrame(ms)
        ax = d.plot(
            title=f"Not Optimized, mean of mem used in {10} runs = {sum(ms)/len(ms)} and max mem usage = {max(ms)}in 10K rows"
        ).figure.savefig("10_loop-10k_bert_NOT_optimized_v2.png")
        # 10k = 3min 13s   optimized
        # 10k = 3 min 37s  NOT optimized

    def test_data_batching(self):
        test = PandasTests().test_py_arrow
        ms = []
        data = {"text": []}
        for i in range(10000):
            data["text"].append("Hello WOrld I like RAM")
        d = pd.DataFrame(data)
        print("GONNA PREDICT!")
        df = nlu.load("ner").predict(d)

        for c in df.columns:
            print(df)

    def test_pyarrow_memory(self):
        test = PandasTests().test_py_arrow
        ms = []
        for i in range(10):
            m = memory_usage((test))
            ms.append(sum(m))
            d = pd.DataFrame(m)
        d = pd.DataFrame(ms)
        ax = d.plot(
            title=f"Not Optimized, mean of mem used in {10} runs = {sum(ms)/len(ms)} and max mem usage = {max(ms)}in 10K rows"
        ).figure.savefig("10_loop-10k_bert_NOT_optimized_v2.png")
        # 10k = 3min 13s   optimized
        # 10k = 3 min 37s  NOT optimized

    def test_py_arrow(self):
        pipe = nlu.load("bert", verbose=True)
        # data = self.load_pandas_dataset()
        data = pd.read_csv(
            "../../tests/datasets/en_lang_filtered_sample.csv"
        )[0:1000]
        big_df = data.append(data)
        for i in range(10):
            big_df = big_df.append(big_df)
        big_df
        df = pipe.predict(big_df[:10000], output_level="document")

        for c in df.columns:
            print(df[c])

    def test_modin(self):
        # ## works with RAY and DASK backends
        df_path = "../../../../tests/datasets/covid/covid19_tweets.csv"
        pdf = pd.read_csv(df_path).iloc[:10]
        secrets_json_path = "../../../../tests/nlu_hc_tests/spark_nlp_for_healthcare.json"

        # test 1 series chunk
        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical resolve.icd10pcs',verbose=True).predict(pdf.text.iloc[0], output_level='chunk')
        # for os_components in res.columns:print(res[os_components])

        # Test longer series chunk
        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical resolve.icd10pcs',verbose=True).predict(pdf.text.iloc[0:10], output_level='chunk')

        # Test df with text col chunk

        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical', verbose=True).predict(pdf.text.iloc[:10], output_level='document')
        # for os_components in res.columns:print(res[os_components])

        # en.resolve_chunk.icd10cm.clinical
        res = (
            nlu.auth(secrets_json_path)
            .load("en.resolve_chunk.icd10cm.clinical", verbose=True)
            .predict(pdf.text[0:7], output_level="chunk")
        )
        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical resolve.icd10pcs',verbose=True).predict(pdf.text[0:7], output_level='chunk')
        for c in res.columns:
            print(res[c])

        # res = nlu.auth(secrets_json_path).load('med_ner.jsl.wip.clinical resolve.icd10pcs',verbose=True).predict(pdf.text[0:7], output_level='sentence')
        # for os_components in res.columns:print(res[os_components])

    def load_pandas_dataset(self):
        output_file_name = "news_category_test.csv"
        output_folder = "classifier_dl/"
        data_url = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/news_Category/news_category_test.csv"
        return pd.read_csv(
            t.download_dataset(data_url, output_file_name, output_folder)
        ).iloc[0:100]


if __name__ == "__main__":

    PandasTests().test_entities_config()
