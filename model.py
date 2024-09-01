import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, dimModel: int, vocabSize: int):
        super().__init__()
        self.dimModel = dimModel
        self.vocabSize = vocabSize
        self.embedding = nn.Embedding(vocabSize, dimModel)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dimModel)

class PositionalEmbedding(nn.Module):
    def __init__(self, dimModel: int, seqLen: int, dropout: float) -> None:
        super().__init__()
        self.dimModel = dimModel
        self.seqLen = seqLen
        
        # Create a dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape [sequence length, dimension of the model]
        positionalEncoding = torch.zeros(seqLen, dimModel)
        # Create a vector of shape [sequence length]
        position = torch.arange(0, seqLen, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dimModel, 2).float() * (-math.log(10000) / dimModel))
        
        # Apply sine and cosine functions
        positionalEncoding[:, 0::2] = torch.sin(position * divTerm)
        positionalEncoding[:, 1::2] = torch.cos(position * divTerm)
        
        positionalEncoding = positionalEncoding.unsqueeze(0)
        
        self.register_buffer('positionalEncoding', positionalEncoding)

    def forward(self, x):
        # Add positional encoding to input x
        x = x + self.positionalEncoding[:, :x.shape[1], :]
        return self.dropout(x)
        