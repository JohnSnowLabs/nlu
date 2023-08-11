import unittest
import nlu
class ConvNextOcrTest(unittest.TestCase):
    def test_img_classification(self):
        img_path = 'tests\\datasets\\ocr\\images\\teapot.jpg'
        p = nlu.load('en.classify_image.convnext.tiny',verbose=True)
        dfs = p.predict(img_path)
        print(dfs['classified_image_results'])

if __name__ == '__main__':
    unittest.main()

