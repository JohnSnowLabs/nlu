import unittest

from nlu import *


class VitTest(unittest.TestCase):
    def test_vit_model(self):
        df = nlu.load("en.classify_image.base_patch16_224").predict([r'/media/ckl/dump/Documents/freelance/MOST_RECENT/jsl/nlu/nlu4realgit3/tests/datasets/ocr/vit/general_images/images/'])
        df = nlu.load("en.classify_image.base_patch16_224").predict([r'/media/ckl/dump/Documents/freelance/MOST_RECENT/jsl/nlu/nlu4realgit3/tests/datasets/ocr/vit/general_images/images/'])
        print(df)
    def test_swin_model(self):
        pipe = nlu.load("en.classify_image.swin.tiny").predict([r'/home/cll/Documents/jsl/nlu4realgit3/tests/datasets/ocr/vit/ox.jpg'])
        print(pipe)


if __name__ == "__main__":
    unittest.main()



