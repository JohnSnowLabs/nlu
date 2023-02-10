import unittest

from nlu import *


class VitTest(unittest.TestCase):
    def test_vit_model(self):
        pipe = nlu.load("en.vit").predict([r'C:\Users\Admin\Documents\GitHub\nlu\tests\datasets\ocr\vit\ox.jpg'])
        print(pipe)
    def test_swin_model(self):
        pipe = nlu.load("en.classify_image.swin.tiny").predict([r'/home/cll/Documents/jsl/nlu4realgit3/tests/datasets/ocr/vit/ox.jpg'])
        print(pipe)


if __name__ == "__main__":
    unittest.main()



