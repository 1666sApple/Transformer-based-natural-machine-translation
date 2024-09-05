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
    
    train = ds_raw[:trainSize]
    val = ds_raw[trainSize:trainSize+valSize]
    test = ds_raw[trainSize+valSize:]
    
    ds_train = BilingualDataset(train, srcTokenizer, targetTokenizer, config['srcLang'], config['targetLang'], config['seqLen'])
    ds_val = BilingualDataset(val, srcTokenizer, targetTokenizer, config['srcLang'], config['targetLang'], config['seqLen'])
    ds_test = BilingualDataset(test, srcTokenizer, targetTokenizer, config['srcLang'], config['targetLang'], config['seqLen'])
    
    srcMaxLength = 0
    targetMaxLength = 0
    
    for item in ds_raw:
        srcID = srcTokenizer.encode(item['translation'][config['srcLang']]).ids
        targetID = targetTokenizer.encode(item['translation'][config['targetLang']]).ids
        srcMaxLength = max(srcMaxLength, len(srcID))
        targetMaxLength = (max(targetMaxLength), len(targetID))
        
    train_dataloader = DataLoader(ds_train, batch_size=config['batchSize'], shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=1, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader, srcTokenizer, targetTokenizer

def getModel(config, srcVocalLen, targetVocalLen):
    model = buildTransformer(srcVocalLen, targetVocalLen, config["seqLen"], config["seqLen"], dimModel=config['dimModel'])
    return model
