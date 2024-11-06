import sys
import os
# sys.path.append(os.getcwd())
import unittest
import nlu

os.environ["PYTHONPATH"] = "F:/Work/repos/nlu_new/nlu"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from johnsnowlabs import nlp, visual

# nlp.install(json_license_path="license.json")

nlp.start()

class TestE5SentenceEmbeddings(unittest.TestCase):
    def test_e5_embeds(self):
        res = nlu.load("en.embed_sentence.e5_small", verbose=True).predict(
            "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
            output_level="document"
        )
        for c in res:
            print(res[c])

        res = nlu.load("en.embed_sentence.e5_base", verbose=True).predict(
            "query: how much protein should a female eat",
            output_level="document"
        )

        for c in res:
            print(res[c])


if __name__ == "__main__":
    unittest.main()
