import unittest
import os


#os.environ['PYSPARK_PYTHON'] = '/home/ckl/anaconda3/bin/python3'
#os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/ckl/anaconda3/bin/python3'



class AsrTestCase(unittest.TestCase):
    def test_wav2vec(self):
        import nlu
        p = nlu.load('en.speech2text.wav2vec2.v2_base_960h',verbose=True)
        FILE_PATH = os.path.normpath(r"tests/datasets/audio/asr/ngm_12484_01067234848.wav")

        print("Got p ",p)
        df = p.predict(FILE_PATH)
        print(df)
        df = p.predict([FILE_PATH,FILE_PATH])
        print(df)


    def test_hubert(self):
        import nlu
        p = nlu.load('en.speech2text.hubert.large_ls960',verbose=True)
        FILE_PATH = os.path.normpath(r"tests/datasets/audio/asr/ngm_12484_01067234848.wav")

        print("Got p ",p)
        df = p.predict(FILE_PATH)
        print(df)
        df = p.predict([FILE_PATH,FILE_PATH])
        print(df)
    def test_whisper(self):
        import nlu
        p = nlu.load('xx.speech2text.whisper.tiny',verbose=True)
        FILE_PATH = os.path.normpath(r"tests/datasets/audio/asr/ngm_12484_01067234848.wav")

        print("Got p ",p)
        df = p.predict(FILE_PATH)
        print(df)
        df = p.predict([FILE_PATH,FILE_PATH])
        print(df)

if __name__ == '__main__':
    unittest.main()
