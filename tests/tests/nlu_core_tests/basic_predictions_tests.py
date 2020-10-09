import unittest
from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
from nlu import *


class TestNameSpace(unittest.TestCase):


    def test_pdf_column_prediction(self):
        pdf = get_sample_pdf()
        res = nlu.load('sentiment',verbose=True).predict(pdf['text'], output_level='sentence')
        # res = nlu.load('bert',verbose=True).predict('@Your life is the sum of a remainder of an unbalanced equation inherent to the programming of the matrix. You are the eventuality of an anomaly, which despite my sincerest efforts I have been unable to eliminate from what is otherwise a harmony of mathematical precision. While it remains a burden assiduously avoided, it is not unexpected, and thus not beyond a measure of control. Which has led you, inexorably, here.', output_level='sentence')

        print(res)
        print(res['sentiment'])

        print(res.dtypes)


    def test_pdf_column_prediction_with_non_text_named_column(self):
        pdf = get_sample_pdf()
        pdf['not_text'] = pdf['text']
        res = nlu.load('tokenize',verbose=True).predict(pdf['not_text'], output_level='sentence')
        # res = nlu.load('bert',verbose=True).predict('@Your life is the sum of a remainder of an unbalanced equation inherent to the programming of the matrix. You are the eventuality of an anomaly, which despite my sincerest efforts I have been unable to eliminate from what is otherwise a harmony of mathematical precision. While it remains a burden assiduously avoided, it is not unexpected, and thus not beyond a measure of control. Which has led you, inexorably, here.', output_level='sentence')

        print(res)
        print(res.columns)

        print(res.dtypes)


    def test_pdf_prediction_with_additional_cols(self):
        # TODO case if input column names overlap with output column names not handeld!
        pdf = get_sample_pdf_with_extra_cols()
        res = nlu.load('pos',verbose=True).predict(pdf)
        # res = nlu.load('bert',verbose=True).predict('@Your life is the sum of a remainder of an unbalanced equation inherent to the programming of the matrix. You are the eventuality of an anomaly, which despite my sincerest efforts I have been unable to eliminate from what is otherwise a harmony of mathematical precision. While it remains a burden assiduously avoided, it is not unexpected, and thus not beyond a measure of control. Which has led you, inexorably, here.', output_level='sentence')

        print(res)
        print(res['pos'])

        print(res.dtypes)

    def test_auto_text_column_infer(self):
        #If no text column exists, NLU will try to use the first column of DF as text colunm
        # This make predicting on dfs easy, by passing a slice where the first element is the prediction col
        # i.e .predict(df[['coment', 'label','some other feature']])
        # This minimizes memory footprint and also saves a line of code
        # TODO case if input column names overlap with output column names not handeld!
        pdf = get_sample_pdf_with_extra_cols()
        pdf['text_name'] = pdf['text']
        # res = nlu.load('tokenize',verbose=True).predict(pdf[['text_name','random_feature1']])
        # # res = nlu.load('bert',verbose=True).predict('@Your life is the sum of a remainder of an unbalanced equation inherent to the programming of the matrix. You are the eventuality of an anomaly, which despite my sincerest efforts I have been unable to eliminate from what is otherwise a harmony of mathematical precision. While it remains a burden assiduously avoided, it is not unexpected, and thus not beyond a measure of control. Which has led you, inexorably, here.', output_level='sentence')
        #
        # print(res.columns)
        # print(res)

        res = nlu.load('tokenize',verbose=True).predict(pdf[['text_name','random_feature1','random_feature2']])
        print(res.columns)
        print(res)

        print(res.dtypes)

    def test_bad_data_input(self):
        pdf = get_sample_pdf_with_no_text_col()
        res = nlu.load('sentiment',verbose=True).predict(pdf, output_level='sentence')
        # res = nlu.load('bert',verbose=True).predict('@Your life is the sum of a remainder of an unbalanced equation inherent to the programming of the matrix. You are the eventuality of an anomaly, which despite my sincerest efforts I have been unable to eliminate from what is otherwise a harmony of mathematical precision. While it remains a burden assiduously avoided, it is not unexpected, and thus not beyond a measure of control. Which has led you, inexorably, here.', output_level='sentence')

        print(res)


    def test_spark_dataframe_input(self):
        sdf = get_sample_spark_dataframe()
        res = nlu.load('sentiment',verbose=True).predict(sdf, output_level='sentence')
        # res = nlu.load('bert',verbose=True).predict('@Your life is the sum of a remainder of an unbalanced equation inherent to the programming of the matrix. You are the eventuality of an anomaly, which despite my sincerest efforts I have been unable to eliminate from what is otherwise a harmony of mathematical precision. While it remains a burden assiduously avoided, it is not unexpected, and thus not beyond a measure of control. Which has led you, inexorably, here.', output_level='sentence')

        print(res)

    def test_bad_component_reference(self):
        sdf = get_sample_spark_dataframe()
        res = nlu.load('asdasj.asdas',verbose=True).predict(sdf, output_level='sentence')
        # res = nlu.load('bert',verbose=True).predict('@Your life is the sum of a remainder of an unbalanced equation inherent to the programming of the matrix. You are the eventuality of an anomaly, which despite my sincerest efforts I have been unable to eliminate from what is otherwise a harmony of mathematical precision. While it remains a burden assiduously avoided, it is not unexpected, and thus not beyond a measure of control. Which has led you, inexorably, here.', output_level='sentence')

        print(res)

if __name__ == '__main__':
    unittest.main()

