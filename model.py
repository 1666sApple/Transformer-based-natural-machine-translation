import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, dimModel: int, vocabSize: int):
        super().__init__()
        self.dimModel = dimModel
        self.embedding = nn.Embedding(vocabSize, dimModel)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dimModel)

class PositionalEmbedding(nn.Module):
    def __init__(self, dimModel: int, seqLen: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positionalEncoding = torch.zeros(seqLen, dimModel)
        position = torch.arange(0, seqLen, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dimModel, 2).float() * (-math.log(10000) / dimModel))
        positionalEncoding[:, 0::2] = torch.sin(position * divTerm)
        positionalEncoding[:, 1::2] = torch.cos(position * divTerm)
        positionalEncoding = positionalEncoding.unsqueeze(0)
        self.register_buffer('positionalEncoding', positionalEncoding)

    def forward(self, x):
        x = x + self.positionalEncoding[:, :x.shape[1], :]
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, dimModel: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dimModel, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, dimModel)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dimModel: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dimModel = dimModel
        self.heads = heads
        self.d_k = dimModel // heads
        self.w_q = nn.Linear(dimModel, dimModel)
        self.w_k = nn.Linear(dimModel, dimModel)
        self.w_v = nn.Linear(dimModel, dimModel)
        self.w_o = nn.Linear(dimModel, dimModel)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attentionScores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attentionScores = attentionScores.masked_fill(mask == 0, -1e9)
        attentionWeights = torch.softmax(attentionScores, dim=-1)
        if dropout is not None:
            attentionWeights = dropout(attentionWeights)
        output = attentionWeights @ value
        return output, attentionWeights

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)
        x, self.attentionScores = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.dimModel)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, features: int, selfAttentionBlock: MultiheadAttentionBlock, feedForwardBlock: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.selfAttentionBlock = selfAttentionBlock
        self.feedForwardBlock = feedForwardBlock
        self.residualConnection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, srcMask):
        x = self.residualConnection[0](x, lambda x: self.selfAttentionBlock(x, x, x, srcMask))
        x = self.residualConnection[1](x, self.feedForwardBlock)
        return x

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features: int, selfAttentionBlock: MultiheadAttentionBlock, crossAttentionBlock: MultiheadAttentionBlock, feedForwardBlock: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.selfAttentionBlock = selfAttentionBlock
        self.crossAttentionBlock = crossAttentionBlock
        self.feedForwardBlock = feedForwardBlock
        self.residualConnection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
        
    def forward(self, x, encoderOutput, srcMask, targetMask):
        x = self.residualConnection[0](x, lambda x: self.selfAttentionBlock(x, x, x, targetMask))
        x = self.residualConnection[1](x, lambda x: self.crossAttentionBlock(x, encoderOutput, encoderOutput, srcMask))
        x = self.residualConnection[2](x, self.feedForwardBlock)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, encoderOutput, srcMask, targetMask):
        for layer in self.layers:
            x = layer(x, encoderOutput, srcMask, targetMask)
        return self.norm(x)
