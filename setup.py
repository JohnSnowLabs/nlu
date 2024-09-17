from codecs import open
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the version from file
with open(path.join(here, 'VERSION')) as version_file:
    version = f"{version_file.read().strip()}"

REQUIRED_PKGS = [
    'spark-nlp>=5.0.2',
    'numpy',
    'pyarrow>=0.16.0',
    'pandas>=1.3.5',
    'dataclasses'
]

setup(

    name='nlu',

    version=version,

    description='John Snow Labs NLU provides state of the art algorithms for NLP&NLU with 20000+ of pretrained models in 200+ languages. It enables swift and simple development and research with its powerful Pythonic and Keras inspired API. It is powerd by John Snow Labs powerful Spark NLP library.',

    long_description=long_description,
    install_requires=REQUIRED_PKGS,

    long_description_content_type='text/markdown',

    url='http://nlu.johnsnowlabs.com',
    author='John Snow Labs',
    author_email='christian@johnsnowlabs.com',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='NLP spark development NLU ',

    packages=find_packages(exclude=['test*', 'tmp*']),
    include_package_data=True
)

