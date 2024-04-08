import unittest

from nlu import *

import os


class TestOpenAICompletion(unittest.TestCase):
    def test_openai_completion(self):

        pipe = nlu.load("openai.completion", apple_silicon=True)

        pipe['openai_completion'].setModel('text-davinci-003')
        pipe['openai_completion'].setMaxTokens(50)

        res = pipe.predict(
            ["Generate a restaurant review.", "Write a review for a local eatery.", "Create a JSON with a review"],
            output_level='document')

        for c in res:
            print(res[c])

if __name__ == "__main__":
    unittest.main()
