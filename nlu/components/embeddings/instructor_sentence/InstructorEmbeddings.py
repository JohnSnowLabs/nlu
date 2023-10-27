from sparknlp.annotator import InstructorEmbeddings


class Instructor:
    @staticmethod
    def get_default_model():
        return InstructorEmbeddings.pretrained() \
            .setInstruction("Instruction here: ") \
            .setInputCols(["documents"]) \
            .setOutputCol("instructor")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return InstructorEmbeddings.pretrained(name,language,bucket) \
            .setInstruction("Instruction here: ") \
            .setInputCols(["documents"]) \
            .setOutputCol("instructor")



