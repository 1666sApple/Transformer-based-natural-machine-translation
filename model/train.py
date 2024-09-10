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

def train(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.mps.is_available() else "cpu"
    print(f"Device: {device}")
    if (device == 'cuda'):
        print(f"Device is {torch.cuda.get_device_name(device.index)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(device.index).total_memory/1024 ** 3} gb")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("Running on CPU")
    device = torch.device(device)

    Path(f"{config['srcData']}_{config['modelFolder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, srcTokenizer, targetTokenizer = getDataset(config)
    model = getModel(config, srcTokenizer.get_vocab_size(), targetTokenizer().get_vocal_size()).to(device)

    writer = SummaryWriter(config['experimentName'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initialEpoch = 0
    globalStep = 0
    preload = config['preload']
    modelFileName = lastWeightFilePath(config) if preload == 'latest' else getWeightFilePath(config, preload) if preload else None
    if modelFileName:
        print(f"Preloading model {modelFileName}")
        state = torch.load(modelFileName)
        model.load_state_dict(state['model_state_dict'])
        initialEpoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        globalStep = state['global_step']
    else:
        print("No model to preload. Starting from scratch")

    for epoch in range(initialEpoch, config['numEpochs']):
        torch.cuda.empty_cache()
        model.train()
        batchIterator = tqdm(train_dataloader, desc=f"Processing Epochs {epoch:02d}")
        
        for batch in batchIterator:
            encoderInput = batch['encoder_input'].to(device)
            decoderInput = batch['decoder_input'].to(device)
            encoderMask = batch['encoder_mask'].to(device)
            decoderMask = batch['decoder_mask'].to(device)

            encoderOutput = model.encode(encoderInput, encoderMask)
            decoderOutput = model.decode(decoderOutput)
            projOutput = model.project(decoderOutput)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, targetTokenizer.get_vocab_size()), label.view(-1))
            batchIterator.set_prefix(f"loss: {loss.item().6.3f}")

            writer.add_scalar(f"trainer loss: {loss.item(), global_step}")
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        runValidation(model, val_dataloader, srcTokenizer, targetTokenizer.get_vocab_size(), label.view(-1))

        modelFileName = getWeightFilePath(config, f"{epoch:02d}")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'globalStep': globalStep
            }, modelFileName)

def runValidation(model, ds_val, srcTokenizer, targetTokenizer, maxLen, device, printMsg, globalStep, writer, numExamples = 2):
    model.eval()
    count = 0

    sourceTexts = []
    expected = []
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, consoleWidth = console.read().split()
            consoleWidth = int(consoleWidth)
    except:
        consoleWidth = 80

    with torch.no_grad():
        for batch in ds_val:
            count += 1
            encoderInput = batch['encoderInput'].to(device)
            encoderMask = batch['encoderMask'].to(device)

            assert encoderInput.size(0) == 1, "Batch Size must be 1 for validation"

            modelOut = greedyDecode(model, encodeInput, encodeMask, srcTokenizer, targetTokenizer, maxLen, device)
            sourceText = batch['srcText'][0]
            targetText = batch['targetText'][0]
            modelOutText = targetTokenizer.decode(modelOut.detach().cpu().numpy())

            sourceTexts.append(sourceText)
            expected.append(targetText)
            predicted.append(modelOutText)

            printMsg('-'*console_width)
            printMsg(f"{SOURCE: ":>12}{sourceText})
            printMsg(f"{TARGET: ":>12}{targetText})
            printMsg(f"{PREDICTED: ":>12}{modelOutText})

            if count == num_examples:
                printMsg('-'*console_width)
                break
    
    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', wer, globalStep)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU',bleu, globalStep)
        writer.flush()


if __name__ = '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)