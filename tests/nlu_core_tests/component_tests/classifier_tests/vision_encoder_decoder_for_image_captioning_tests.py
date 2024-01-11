import unittest

from nlu import *


class VisionEncoderDecoderTest(unittest.TestCase):
    def test_image_captioning_vit_gpt2_model(self):
        df = nlu.load("en.classify_image.image_captioning_vit_gpt2").predict([r'./../../../datasets/ocr/vit/general_images/images'])
        print(df)

if __name__ == "__main__":
    unittest.main()



