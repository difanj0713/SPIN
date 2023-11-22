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