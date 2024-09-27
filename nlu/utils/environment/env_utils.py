import logging
import os
import platform
import subprocess
import sys

logger = logging.getLogger('nlu')


def get_pyspark_version():
    import pyspark
    return pyspark.version.__version__


def get_pyspark_major_and_minor():
    import pyspark
    return pyspark.version.__version__.split('.')[:2]


def is_env_pyspark_3_1():
    v = get_pyspark_major_and_minor()
    if v[0] == '3' and v[1] == '1':
        return True
    return False

def is_env_pyspark_3_2():
    v = get_pyspark_major_and_minor()
    if v[0] == '3' and v[1] == '2':
        return True
    return False


def is_env_pyspark_3_0():
    v = get_pyspark_major_and_minor()
    if v[0] == '3' and v[1] == '0':
        return True
    return False


def is_env_pyspark_2_4():
    v = get_pyspark_major_and_minor()
    if v[0] == '2' and v[1] == '4':
        return True
    return False


def is_env_pyspark_2_3():
    v = get_pyspark_major_and_minor()
    if v[0] == '2' and v[1] == '3':
        return True
    return False

def is_env_pyspark_2_x():
    v = get_pyspark_major_and_minor()
    return int(v[0]) <= 2

def is_env_pyspark_3_x():
    v = get_pyspark_major_and_minor()
    return int(v[0]) == 3




def check_pyspark_install():
    try:
        from pyspark.sql import SparkSession
        try:
            import sparknlp
            v = sparknlp.start().version
            spark_major = int(v.split('.')[0])
            if spark_major >= 3:
                raise Exception()
        except:
            print(
                f"Detected pyspark version={v} Which is >=3.X\nPlease run '!pip install pyspark==2.4.7' or install any pyspark>=2.4.0 and pyspark<3")
            return False
    except:
        print(
            "No Pyspark installed!\nPlease run '!pip install pyspark==2.4.7' or install any pyspark>=2.4.0 with pyspark<3")
        return False
    return True

def try_import_streamlit():
    try:
        import streamlit as st
    except  ImportError:
        print("You need to install Streamlit to run this functionality.")


def is_running_in_databricks_runtime():
    """ Check if the currently running Python Process is running in Databricks runtime or not
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def install_and_import_package(pkg_name, version='', import_name=''):
    """ Install Spark-NLP-Healthcare PyPI Package in current environment if it cannot be imported and license
    provided """
    import importlib
    try:
        if import_name == '':
            importlib.import_module(pkg_name)
        else:
            importlib.import_module(import_name)
    except ImportError:
        import pip
        if version == '':
            logger.info(f"{pkg_name} could not be imported. Running 'pip install {pkg_name}'...")
        else:
            logger.info(f"{pkg_name} could not be imported. Running 'pip install {pkg_name}=={version}'...")
        pip_major_version = int(pip.__version__.split('.')[0])
        if pip_major_version in [10, 18, 19, 20]:
            # for these versions pip module does not support installing, we install via OS command straight into pip
            # module
            py_path = sys.executable
            if version == '':
                os.system(f'{py_path} -m pip install {pkg_name}')
            else:
                os.system(f'{py_path} -m pip install {pkg_name}=={version}')
        else:
            if version == '':
                pip.main(['install', f'{pkg_name}'])
            else:
                pip.main(['install', f'{pkg_name}=={version}'])
    finally:
        import site
        from importlib import reload
        reload(site)
        # import name is not always the same name as pkg_name we want to import, so it must be specified via import name
        if import_name != '':
            globals()[import_name] = importlib.import_module(import_name)
        else:
            globals()[pkg_name] = importlib.import_module(pkg_name)



def try_import_pyspark_in_streamlit():
    """Try importing Pyspark or display warn message in streamlit"""
    try:
        import pyspark
        from pyspark.sql import SparkSession
    except:
        print("You need Pyspark installed to run NLU. Run <pip install pyspark==3.0.2>")
        try:
            import streamlit as st
            st.error(
                "You need Pyspark, Sklearn, Pyplot, Pandas, Numpy installed to run this app. Run <pip install pyspark==3.0.2 sklearn pyplot numpy pandas>")
        except:
            return False
        return False
    return True

def try_import_spark_nlp():
    """Try importing Spark NLP"""
    try:
        import sparknlp
    except:
        print("You need Spark NLP to run NLU. run pip install spark-nlp")
        return False
    return True


def is_m1():
    """Checks whether the current processor is an Apple M1 chip."""
    platform_info = platform.uname()
    if platform_info.system == "Darwin" and platform_info.machine == "arm64":
        sys_info_query = "sysctl -n machdep.cpu.brand_string"
        result = subprocess.run(sys_info_query.split(), capture_output=True)
        cpu = result.stdout.decode().strip()
        if cpu == "Apple M1":
            return True
        else:
            logger.info(
                (
                    "Currently, only the standard M1 processor has experimental "
                    "support.\nOther derivations like M1 Pro/Max/Ultra will be "
                    "supported in the future."
                )
            )
            return False
    else:
        return False
