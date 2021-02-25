from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
import unittest
# from tests.test_utils import get_sample_pdf_with_labels, get_sample_pdf, get_sample_sdf, get_sample_pdf_with_extra_cols, get_sample_pdf_with_no_text_col ,get_sample_spark_dataframe
# from nlu import *


class TestAuthentification(unittest.TestCase):


    def test_bad_component_reference(self):


        def install_and_import_healthcare(JSL_SECRET):
            # pip install  spark-nlp-jsl==2.7.3 --extra-index-url https://pypi.johnsnowlabs.com/2.7.3-3f5059a2258ea6585a0bd745ca84dac427bca70c --upgrade
            """ Install Spark-NLP-Healthcare PyPI Package in current enviroment if it cannot be imported and liscense provided"""
            import importlib
            try:
                importlib.import_module('sparknlp_jsl')
            except ImportError:
                import pip
                print("Spark NLP Healthcare could not be imported. Installing latest spark-nlp-jsl PyPI package via pip...")
                pip.main(['install', 'spark-nlp-jsl==2.7.3', '--extra-index-url', f'https://pypi.johnsnowlabs.com/{JSL_SECRET}'])
            finally:
                import site
                from importlib import reload
                reload(site)
                globals()['sparknlp_jsl'] = importlib.import_module('sparknlp_jsl')

        def authenticate_enviroment(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY):
            """Set Secret environ variables for Spark Context"""
            import os
            os.environ['SPARK_NLP_LICENSE'] = SPARK_NLP_LICENSE
            os.environ['AWS_ACCESS_KEY_ID']= AWS_ACCESS_KEY_ID
            os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

        def get_authenticated_spark(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET):
            """
            Authenticates enviroment if not already done so and returns Spark Context with Healthcare Jar loaded
            0. If no Spark-NLP-Healthcare, install it via PyPi
            1. If not auth, run authenticate_enviroment()

            """
            install_and_import_healthcare(JSL_SECRET)
            authenticate_enviroment(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)

            import sparknlp_jsl
            return sparknlp_jsl.start(JSL_SECRET)


        # # 2.7.3 AUTH

        SPARK_NLP_LICENSE = ''
        AWS_ACCESS_KEY_ID = ''
        AWS_SECRET_ACCESS_KEY = ''
        JSL_SECRET = ''
        get_authenticated_spark(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET)
        import sparknlp_jsl
        from sparknlp_jsl.annotator import ChunkEntityResolverModel

        m = ChunkEntityResolverModel.pretrained('chunkresolve_athena_conditions_healthcare','en','clinical/models')
        print(m)
if __name__ == '__main__':
    unittest.main()

