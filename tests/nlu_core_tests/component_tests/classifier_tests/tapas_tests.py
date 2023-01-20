import unittest
import sparknlp
import librosa as librosa
from sparknlp.base import *
from sparknlp.annotator import *
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
import pyspark.sql.functions as F
import sparknlp
import sparknlp
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
import os


os.environ['PYSPARK_PYTHON'] = '/home/ckl/anaconda3/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/ckl/anaconda3/bin/python3'



class TapasCase(unittest.TestCase):
    def test_tapas(self):
        """
        json_path = 'my/json_file.json'
        1. pipe.predict(json_path)
        2. pipe.predict([json_path,json_path,json_path])

        json_string = "chungos muxos"
        3. pipe.predict(json_string)
        4. pipe.predict([json_string,json_string,])


        :return:
        """
        import nlu
        # p = nlu.load('en.tapas.wip',verbose=True)
        spark = sparknlp.start()
        json_data = """
{
  "header": ["name", "money", "age"],
  "rows": [
    ["Donald Trump", "$100,000,000", "75"],
    ["Elon Musk", "$20,000,000,000,000", "55"]
  ]
}
"""

# {"header": ["name","money","age"], "rows": [["Donald Trump","$100,1000,000","75"],["Elon Musk","$100,1000,000,000","55"]] }
# {"header": ["name","money","age"], "rows": [["Donald Trump","$100,000,000","75"],["Elon Musk", "$20,000,000,000,000", "55"]]}


        queries = [
    "Who earns less than 200,000,000?",
    "Who earns 100,000,000?",
    "How much money has Donald Trump?",
    "How old are they?",
]
        data = spark.createDataFrame([
            [json_data, " ".join(queries)]
        ]).toDF("table_json", "questions")
        csv_path = '/media/ckl/dump/Documents/freelance/MOST_RECENT/jsl/nlu/nlu4realgit3/tests/datasets/healthcare/sample_ADE_dataset.csv'
        csv_data = pd.read_csv(csv_path)
        document_assembler = MultiDocumentAssembler() \
            .setInputCols("table_json", "questions") \
            .setOutputCols("document_table", "document_questions")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document_questions"]) \
            .setOutputCol("questions")
        table_assembler = TableAssembler() \
            .setInputCols(["document_table"]) \
            .setOutputCol("table") \
            .setInputFormat('csv')
        # tapas = TapasForQuestionAnswering \
        #     .pretrained("table_qa_tapas_base_finetuned_wtq", "en") \
        #     .setInputCols(["questions", "table"]) \
        #     .setOutputCol("answers")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            table_assembler,
            # tapas
        ])


        model = pipeline.fit(data)
        model \
            .transform(data) \
            .selectExpr("explode(answers) AS answer") \
            .select("answer") \
            .show(truncate=False)



    def test_tapas_nlu_json_string(self):
        """
        Like QA DataFormat for Question Answering .
        Take in 1 CSV-Ish data-object + 1 Question-Store-Object.
        Question-Store-Object is either Str, or array of Str where each element is a question to be asked on the CSV object
        nlu.load(tapas).predict((tabular_data, question_data))
        tabular_data  may be Pandas DF or Tabular Data String (JSON/CSV)
        question_data may be a string or a list of Strings
        nlu.load(tapas).predict((tabular_data, 'How old is the average employee?'))
        nlu.load(tapas).predict((company_df, ['How old is the average employee?', 'How many people work in the IT deparment?']))








        # One column must be question, everything else is context.
        input = /TupleIterable
            with len(input) == 2
            input[0] = table_like
            input[0] = str, Iterable[str]
        p.predict((tabular_data, question(s)))
        p.predict((tabular_data, question(s)))


        # One Key must be question, ewverything else is context
        p.predict(json_string)
        p.predict(csv_string,q)

        p.predict(json_pat,qh)
        p.predict(csv_path,q)


        p.predict('Hello World') # NOT SUPPORTED!
        Metadata Keys : question, aggregation, cell_positions cell_scores
        :return:
        """
        spark = sparknlp.start()
        data_df = pd.DataFrame({'name':['Donald Trump','Elon Musk'], 'money': ['$100,000,000','$20,000,000,000,000'], 'age' : ['75','55'] })
        # {"header": ["name","money","age"], "rows": [["Donald Trump","$100,000,000","75"],["Elon Musk", "$20,000,000,000,000", "55"]]}

        questions = [
            "Who earns less than 200,000,000?",
            "Who earns 100,000,000?",
            "How much money has Donald Trump?",
            "How old are they?",
        ]

        tapas_data = (data_df, questions)
        import nlu
        p = nlu.load('en.tapas.wip')
        res = p.predict(tapas_data)
        print(p)
        for c in res.columns:
            print(res[c])

if __name__ == '__main__':
    unittest.main()

