import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint
import numpy as np
import pandas as pd
import datasets
from imports import *

import argparse
import typing
import os
import re
import sys
import time
import json
import pickle 
from collections import Counter

from transformers import GPT2Tokenizer, GPT2Model
from transformers import RobertaTokenizer, RobertaModel, DistilBertTokenizer, DistilBertModel
#from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score, r2_score, classification_report

import plotly.express as px

import importlib
import warnings
from collections import defaultdict
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def my_train_test_split(dataset_name):
    if dataset_name == "imdb":
        dataset = datasets.load_dataset('imdb', cache_dir='../dataset/IMDb-plain_text')
        text_train = dataset['train']['text']
        label_train = dataset['train']['label']
        text_test = dataset['test']['text']
        label_test = dataset['test']['label']
        text_train, text_val, label_train, label_val = train_test_split(text_train, label_train, test_size=0.20, random_state=42) # val split

    if dataset_name == "edos":
        dataset_path = "../dataset/sexism.csv" # after a simple data wraggler to transform sexist/non-sexist to 0-1 labels
        df = pd.read_csv(dataset_path, encoding='utf-8')

        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
        val_df = df[df['split'] == 'dev']
        train_df = train_df.reindex(train_df['text'].str.len().sort_values(ascending=False).index)
        test_df = test_df.reindex(test_df['text'].str.len().sort_values(ascending=False).index)
        val_df = val_df.reindex(val_df['text'].str.len().sort_values(ascending=False).index)

        text_train = train_df['text'].to_list()
        label_train = np.array(train_df['label'].to_list())
        text_test = test_df['text'].to_list()
        label_test = np.array(test_df['label'].to_list())
        text_val = val_df['text'].to_list()
        label_val = np.array(val_df['label'].to_list())

    if dataset_name == "sst-2":
        dataset = datasets.load_dataset('glue', 'sst2', cache_dir='../dataset/sst-2') # to conform with general fine-tuned approach, we report val performance
        text_train = dataset['train']['sentence']
        label_train = dataset['train']['label']
        text_train, text_val, label_train, label_val = train_test_split(text_train, label_train, test_size=0.03, random_state=42) # val split
        #text_test = dataset['test']['sentence']
        #label_test = dataset['test']['label']
        text_test = dataset['validation']['sentence']
        label_test = dataset['validation']['label']
    
    return text_train, text_val, text_test, label_train, label_val, label_test