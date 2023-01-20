import unittest

from nlu import *


class VitTest(unittest.TestCase):
    def test_vikt_model(self):
        pipe = nlu.load("en.vit").predict([r'C:\Users\Admin\Documents\GitHub\nlu\tests\datasets\ocr\vit\ox.jpg'])
        print(pipe)



if __name__ == "__main__":
    unittest.main()



