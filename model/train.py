import os
import torch
import torch.nn as nn
import torchmetrics
import warnings
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from pathlib import Path

from model.dataset import BilingualDataset, causalMask
from model.transformers import buildTransformer
from model.config import getConfig, getWeightFilePath, lastWeightFilePath
from model.utils import getAllSentences, getOrBuildTokenizer, greedyDecode


def getDataset(config):
    ds_raw = load_dataset(f"{config['srcData']}", f"{config['srcLang']}-{config['targetLang']}", split='train')
    
    srcTokenizer = getOrBuildTokenizer(config, ds_raw, config['srcLang'])
    targetTokenizer = getOrBuildTokenizer(config, ds_raw, config['targetLang'])
    
    trainSize = int(0.8 * len(ds_raw))
    valSize = int(0.1 * len(ds_raw))
    testSize = len(ds_raw) - (trainSize + valSize)