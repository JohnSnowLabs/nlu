import unittest

from nlu import *


class ClipTest(unittest.TestCase):

    def test_clip_model(self):
        clip = nlu.load('en.classify_image.clip_zero_shot')
        candidateLabels = [
            "a photo of a bird",
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a hen",
            "a photo of a hippo",
            "a photo of a palace",
            "a photo of a tractor",
            "a photo of an ostrich",
            "a photo of an ox"]
        clip['clip_for_zero_shot_image_classification'].setCandidateLabels(candidateLabels)

        df = clip.predict('./../../../datasets/ocr/vit/general_images/images')
        print(df)


if __name__ == "__main__":
    unittest.main()



