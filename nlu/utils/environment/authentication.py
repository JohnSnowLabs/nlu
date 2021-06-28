from nlu.utils.environment.env_utils import *







def install_and_import_healthcare(JSL_SECRET):
    """ Install Spark-NLP-Healthcare PyPI Package in current enviroment if it cannot be imported and liscense provided"""
    import importlib
    try:
        importlib.import_module('sparknlp_jsl')
    except ImportError:
        import pip
        print("Spark NLP Healthcare could not be imported. Installing latest spark-nlp-jsl PyPI package via pip...")
        hc_version      = JSL_SECRET.split('-')[0]
        import pyspark
        pip_major_version = int(pip.__version__.split('.')[0])
        if pip_major_version in [10, 18, 19,20]:
            # for these versions pip module does not support installing, we install via OS command.
            os.system(f'pip install spark-nlp-jsl=={hc_version} --extra-index-url https://pypi.johnsnowlabs.com/{JSL_SECRET}')
        else:
            pip.main(['install', f'spark-nlp-jsl=={hc_version}', '--extra-index-url', f'https://pypi.johnsnowlabs.com/{JSL_SECRET}'])
    finally:
        import site
        from importlib import reload
        reload(site)
        globals()['sparknlp_jsl'] = importlib.import_module('sparknlp_jsl')







def authenticate_enviroment(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY):
    """Set Secret environ variables for Spark Context"""
    os.environ['SPARK_NLP_LICENSE'] = SPARK_NLP_LICENSE
    os.environ['AWS_ACCESS_KEY_ID']= AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

def get_authenticated_spark(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, gpu=False,):
    """
    Authenticates enviroment if not already done so and returns Spark Context with Healthcare Jar loaded
    0. If no Spark-NLP-Healthcare, install it via PyPi
    1. If not auth, run authenticate_enviroment()

    """

    authenticate_enviroment(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
    install_and_import_healthcare(JSL_SECRET)


    import sparknlp_jsl
    if is_env_pyspark_2_3() : return sparknlp_jsl.start(JSL_SECRET, spark23=True, gpu=gpu)
    if is_env_pyspark_2_4() : return sparknlp_jsl.start(JSL_SECRET, spark24=True, gpu=gpu)
    if is_env_pyspark_3_0() or is_env_pyspark_3_1()  : return sparknlp_jsl.start(JSL_SECRET, gpu=gpu, public='3.1.0')
    print(f"Current Spark version {get_pyspark_version()} not supported!")
    raise ValueError

def is_authorized_enviroment():
    """Check if auth secrets are set in enviroment"""
    SPARK_NLP_LICENSE = os.getenv('SPARK_NLP_LICENSE')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    return None not in [SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY]
