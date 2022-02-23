import unittest
from nlu import *


class TestChunks(unittest.TestCase):

    def test_chunker(self):
        example_text = ["A person like Jim or Joe",
                        "An organisation like Microsoft or PETA",
                        "A location like Germany",
                        "Anything else like Playstation",
                        "Person consisting of multiple tokens like Angela Merkel or Donald Trump",
                        "Organisations consisting of multiple tokens like JP Morgan",
                        "Locations consisting of multiple tokens like Los Angeles",
                        "Anything else made up of multiple tokens like Super Nintendo", ]
        res = nlu.load('chunk').predict(example_text, output_level='sentence', drop_irrelevant_cols=False,
                                           metadata=True, )
        for c in res.columns:
            print(res[c])


if __name__ == '__main__':
    unittest.main()
