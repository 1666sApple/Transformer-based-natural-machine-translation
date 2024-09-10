import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train import train_model
from model.config import getConfig
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = getConfig()
    train_model(config)