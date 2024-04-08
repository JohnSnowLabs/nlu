import unittest

from nlu import *


class TestInstructorSentenceEmbeddings(unittest.TestCase):
    def test_instructor_embeds_sentence_level(self):
        pipe = nlu.load("en.embed_sentence.instructor_base", verbose=True)
        pipe['instructor_sentence_embeddings@INSTRUCTOR_EMBEDDINGS_1c5e51202650'].setInstruction(
            "Represent the Amazon title for retrieving relevant reviews: ")
        res = pipe.predict("Loved it!  It is Exciting, interesting, and even including information about the space program.",
                           output_level='sentence')

        for c in res:
            print(res[c])

        pipe = nlu.load("en.embed_sentence.instructor_large", verbose=True)
        pipe['instructor_sentence_embeddings@INSTRUCTOR_EMBEDDINGS_46e0451abc97'].setInstruction(
            "Represent the Amazon title for retrieving relevant reviews: ")
        res = pipe.predict("Loved it!  It is Exciting, interesting, and even including information about the space program.",
                           output_level='sentence')

        for c in res:
            print(res[c])

    def test_instructor_embeds_document_level(self):
        pipe = nlu.load("en.embed_sentence.instructor_base", verbose=True)
        pipe['instructor_sentence_embeddings@INSTRUCTOR_EMBEDDINGS_1c5e51202650'].setInstruction(
            "Represent the Amazon title for retrieving relevant reviews: ")
        res = pipe.predict("Loved it!  It is Exciting, interesting, and even including information about the space program.",
                           output_level='document')

        for c in res:
            print(res[c])

        pipe = nlu.load("en.embed_sentence.instructor_large", verbose=True)
        pipe['instructor_sentence_embeddings@INSTRUCTOR_EMBEDDINGS_46e0451abc97'].setInstruction(
            "Represent the Amazon title for retrieving relevant reviews: ")
        res = pipe.predict("Loved it!  It is Exciting, interesting, and even including information about the space program.",
                           output_level='document')

        for c in res:
            print(res[c])

if __name__ == "__main__":
    unittest.main()