import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, dimModel: int, vocabSize: int):
        super().__init__()
        self.dimModel = dimModel
        self.vocabSize = vocabSize
        # Embedding layer
        self.embedding = nn.Embedding(vocabSize, dimModel)
        
    def forward(self, x):
        # Scale the embedding by the square root of the model dimension
        return self.embedding(x) * math.sqrt(self.dimModel)

class PositionalEmbedding(nn.Module):
    def __init__(self, dimModel: int, seqLen: int, dropout: float) -> None:
        super().__init__()
        self.dimModel = dimModel
        self.seqLen = seqLen
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        positionalEncoding = torch.zeros(seqLen, dimModel)
        position = torch.arange(0, seqLen, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dimModel, 2).float() * (-math.log(10000) / dimModel))
        
        # Apply sine to even indices in the positional encoding
        positionalEncoding[:, 0::2] = torch.sin(position * divTerm)
        # Apply cosine to odd indices in the positional encoding
        positionalEncoding[:, 1::2] = torch.cos(position * divTerm)
        
        positionalEncoding = positionalEncoding.unsqueeze(0)
        
        # Register positional encoding as a buffer (non-learnable parameter)
        self.register_buffer('positionalEncoding', positionalEncoding)

    def forward(self, x):
        # Add positional encoding to input and apply dropout
        x = x + self.positionalEncoding[:, :x.shape[1], :]
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter
        self.alpha = nn.Parameter(torch.ones(1))
        # Learnable bias parameter
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Compute mean and standard deviation across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # Normalize and apply learned scale and bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, dimModel: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # First linear layer
        self.linear1 = nn.Linear(dimModel, d_ff)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Second linear layer
        self.linear2 = nn.Linear(d_ff, dimModel)
        
    def forward(self, x):
        # Apply the feed-forward block
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dimModel: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dimModel = dimModel
        self.heads = heads
        assert dimModel % heads == 0
        
        self.d_k = dimModel // heads
        # Linear layers for query, key, and value
        self.w_q = nn.Linear(dimModel, dimModel)
        self.w_k = nn.Linear(dimModel, dimModel)
        self.w_v = nn.Linear(dimModel, dimModel)
        
        # Linear layer for output
        self.w_o = nn.Linear(dimModel, dimModel)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Calculate attention scores
        attentionScores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) 
        
        if mask is not None:
            # Apply mask to attention scores
            attentionScores = attentionScores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights and apply dropout
        attentionWeights = torch.softmax(attentionScores, dim=-1)
        
        if dropout is not None:
            attentionWeights = dropout(attentionWeights)
            
        # Calculate the output as weighted sum of values
        output = attentionWeights @ value
        
        return output, attentionWeights

    def forward(self, q, k, v, mask):
        # Apply linear layers and reshape for multi-head attention
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)
        
        # Compute attention and reshape output
        x, self.attentionScores = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.dimModel)
        return self.w_o(x)
