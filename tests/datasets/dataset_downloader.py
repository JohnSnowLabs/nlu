'''
For each of the trainable models in NLU, download a dataset for using it



1. train.deep_sentence_detector
    train.sentence_detector
2. train.symmetric_spell
3. train.context_spell
    train.spell
4. train.norvig_spell
5. train.unlabeled_dependency_parser
6. train.labeled_dependency_parser
7. train.classifier_dl
    train.classifier
8. train.named_entity_recognizer_dl
    train.ner
9. train.vivekn_sentiment
10.train.sentiment_dl
    train.sentiment
11.train.pos
    train.pos
12.train.multi_classifier

'''
# preprocessing
deep_sentence_detector_data_url = ''
# spell
symmetric_spell_data_url = ''
context_spell_data_url = ''
norvig_spell_data_url = ''
# Dep Parsing
unlabeled_dependency_parser_data_url = ''
labeled_dependency_parser_data_url = ''

# Classifiers
classifier_dl_data_url = ''
named_entity_recognizer_dl_data_url = 'https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/data/ner'
vivekn_sentiment_data_url = ''
sentiment_dl_data_url = ''
pos_data_url = ''
multi_classifier_data_url = ''

'''
# load a model, download a dataset and fit the model on it in 1 line!
nlu.load('train.ner).fit(nlu.Dataset('train.ner'))
model_to_train = 'train_ner
nlu.load(model_to_train).fit(nlu.Dataset(model_to_train))

'''
@dataclass
class DatasetInfor():
    model_name : str
    train_url : str
    test_url : str

class Dataset():
    def __init__(self, nlu_ref):
        self.nlu_ref=nlu_ref
        self.dataset_urls =  {}
        self.dataset_paths = {}


    def resolve_nlu_ref_to_dataset(self,nlu_ref):
        # for a given nlu_ref, returns which dataset url and path should be used


class DatasetDownloader():

dataset_directory = '/home/loan/Documents/freelancework/jsl/nlu/4realnlugit/tests/datasets'

def set_dataset_directory(): pass # configure where datasets to download
def check_if_datset_exists():pass
def get_classifier_dataset():pass
def get

def download_dataset_if_not_present_and_get_path(path):
    '''

    :param path: Name of where the file should be located if it exists
    :return:
    '''

    if not Path(download_path).is_file():
        print("Dataset not found, will download it!")
    urllib.request.urlretrieve(file_url, download_path)
    else:
        print("Dataset already exists")
        # tpdp get path to dataset

def get_datset(model_name):
    '''
    Get a dataset for any of the trainable models in NLU by passing the reference of the trainable model.
    Renames columns accordingly to X and y
    :param model_name:
    :return:
    '''
    switcher = {
        'classifier': get_classifier_dataset(),
        }


import pandas as pd
import os
from pathlib import Path
import urllib.request
file_url = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sarcasm/train-balanced-sarcasm.csv'
download_path = "./sarcasm.csv"



if not Path(download_path).is_file():
    print("File Not found will downloading it!")
    urllib.request.urlretrieve(file_url, download_path)
else:
    print("File already present.")