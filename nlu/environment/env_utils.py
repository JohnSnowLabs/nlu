import os,sys

def get_pyspark_version():
    import pyspark
    return pyspark.version.__version__

def get_pyspark_major_and_minor():
    import pyspark
    return pyspark.version.__version__.split('.')[:2]


def is_env_pyspark_3_1():
    v = get_pyspark_major_and_minor()
    if v[0] == '3' and v[1] == ['1'] : return True
    return False

def is_env_pyspark_3_0():
    v = get_pyspark_major_and_minor()
    if v[0] == '3' and v[1] == ['0'] : return True
    return False


def is_env_pyspark_2_4():
    v = get_pyspark_major_and_minor()
    if v[0] == '2' and v[1] == ['4'] : return True
    return False

def is_env_pyspark_2_3():
    v = get_pyspark_major_and_minor()
    if v[0] == '2' and v[1] == ['3'] : return True
    return False


def check_pyspark_install():
    try :
        from pyspark.sql import SparkSession
        try :
            import sparknlp
            v = sparknlp.start().version
            spark_major = int(v.split('.')[0])
            if spark_major >= 3 :
                raise Exception()
        except :
            print(f"Detected pyspark version={v} Which is >=3.X\nPlease run '!pip install pyspark==2.4.7' or install any pyspark>=2.4.0 and pyspark<3")
            # print(f"Or set nlu.load(version_checks=False). We disadvise from doing so, until Pyspark >=3 is officially supported in 2021.")
            return False
    except :
        print("No Pyspark installed!\nPlease run '!pip install pyspark==2.4.7' or install any pyspark>=2.4.0 with pyspark<3")
        return False
    return True

def check_python_version():
    if float(sys.version[:3]) >= 3.9:
        print("Please use a Python version with version number SMALLER than 3.9")
        print("Python versions equal or higher 3.9 are currently NOT SUPPORTED by NLU")
        return False
    return True

def is_running_in_databricks():
    """ Check if the currently running Python Process is running in Databricks or not
     If any Enviroment Variable name contains 'DATABRICKS' this will return True, otherwise False"""
    for k in os.environ.keys() :
        if 'DATABRICKS' in k :
            return True
    return False
