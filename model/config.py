from pathlib import Path

def getConfig():
    return {
        "batchSize": 8,
        "numEpochs": 20,
        "lr": 1e-4,
        "seqLen": 256,
        "dimModel": 512,
        "srcData": 'opus_books',
        "srcLang": "en",
        "targetLang": "fr",
        "model_folder": "weights",  # Changed to lowercase 'f'
        "modelBaseName": "transformer",
        "preload": None,
        "tokenizerFile": "tokenizer{0}.json",
        "experimentName": "runs/transformer"
    }


def getWeightFilePath(config, epoch: str):
    modelFolder = f"{config['srcData']}_{config['modelFolder']}"
    modelFileName = f"{config['modelBaseName']}{epoch}.pt"
    return str(Path('.') / modelFolder / modelFileName)

def lastWeightFilePath(config):
    modelFolder = f"{config['srcData']}_{config['modelFolder']}"
    modelFileName = f"{config['modelBaseName']}*"
    weightFiles = list(Path(modelFolder).glob(modelFileName))
    if len(weightFiles) == 0:
        return None
    weightFiles.sort()
    return str(weightFiles[-1])