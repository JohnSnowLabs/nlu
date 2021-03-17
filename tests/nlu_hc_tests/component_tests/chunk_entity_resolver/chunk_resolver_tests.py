import unittest
import pandas as pd
import nlu

class ChunkResolverTests(unittest.TestCase):





    def test_chunk_resolver(self):

        SPARK_NLP_LICENSE     = ''
        AWS_ACCESS_KEY_ID     = ''
        AWS_SECRET_ACCESS_KEY = ''
        JSL_SECRET            = ''

        nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        res = nlu.load('en.ner.diseases en.resolve_chunk.snomed.findings', verbose=True).predict(['The patient has cancer and high fever and will die next week.', ' She had a seizure.'], drop_irrelevant_cols=False, metadata=True, return_spark_df=True)


        res.show()
        for c in res.columns:
            print(c)
            res.select(c).show(truncate=False)
        # for c in res :
        #     print(res[c])
        # 
        # print(res)
if __name__ == '__main__':
    ChunkResolverTests().test_entities_config()


