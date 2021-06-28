from sparknlp.annotator import *

class T5:
    @staticmethod
    def nlu_ref_to_task(task):
        nlu_to_task = {
            'check_grammar': 'cola sentence: ',
            'summarize': 'summarize: ',
            'sentiment': 'sst2 sentence:  ',
            'answer_question': 'question:  ',

        }
        if task in nlu_to_task.keys() :return nlu_to_task[task]
        else : return task
    @staticmethod
    def get_default_model():
        return T5Transformer.pretrained() \
        .setInputCols("document") \
        .setOutputCol("T5")

    @staticmethod
    def get_pretrained_model(name, language,bucket=None):
        return T5Transformer.pretrained(name, language,bucket) \
            .setInputCols("document") \
            .setOutputCol("T5")

    @staticmethod
    # Get T5 with task pre-configured
    # Sets task either to somthing known or whatever what detected
    def get_preconfigured_model(name, language,task):

        return T5Transformer.pretrained(name, language) \
            .setInputCols("document") \
            .setOutputCol("T5")\
            .setTask(T5.nlu_ref_to_task(task))





