import torch
import torch.nn as nn
import math

# Input embedding module with scaling by sqrt(dimModel)
class InputEmbedding(nn.Module):
    def __init__(self, dimModel: int, vocabSize: int):
        super().__init__()
        self.dimModel = dimModel
        self.embedding = nn.Embedding(vocabSize, dimModel)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dimModel)

# Positional encoding module with dropout
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
        # Add positional encoding to input tensor and apply dropout
        x = x + self.positionalEncoding[:, :x.shape[1], :]
        return self.dropout(x)

# Layer normalization module
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

# Feed-forward block used in Transformer layers
class FeedForwardBlock(nn.Module):
    def __init__(self, dimModel: int, feedForwardDim: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dimModel, feedForwardDim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedForwardDim, dimModel)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# Multi-head attention block
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
        # Compute scaled dot-product attention
        attentionScores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attentionScores = attentionScores.masked_fill(mask == 0, -1e9)
        attentionWeights = torch.softmax(attentionScores, dim=-1)
        if dropout is not None:
            attentionWeights = dropout(attentionWeights)
        output = attentionWeights @ value
        return output, attentionWeights

    def forward(self, q, k, v, mask):
        # Project inputs to query, key, and value spaces
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # Reshape for multi-head attention
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)
        # Apply attention mechanism
        x, self.attentionScores = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # Concatenate heads and apply final linear transformation
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.dimModel)
        return self.w_o(x)

# Residual connection module with layer normalization and dropout
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Encoder block consisting of self-attention and feed-forward blocks
class EncoderBlock(nn.Module):
    def __init__(self, features: int, selfAttentionBlock: MultiheadAttentionBlock, feedForwardBlock: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.selfAttentionBlock = selfAttentionBlock
        self.feedForwardBlock = feedForwardBlock
        self.residualConnection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, srcMask):
        # Apply self-attention and feed-forward with residual connections
        x = self.residualConnection[0](x, lambda x: self.selfAttentionBlock(x, x, x, srcMask))
        x = self.residualConnection[1](x, self.feedForwardBlock)
        return x

# Encoder consisting of multiple encoder blocks
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, mask):
        # Pass through each encoder block
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Decoder block with self-attention, cross-attention, and feed-forward blocks
class DecoderBlock(nn.Module):
    def __init__(self, features: int, selfAttentionBlock: MultiheadAttentionBlock, crossAttentionBlock: MultiheadAttentionBlock, feedForwardBlock: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.selfAttentionBlock = selfAttentionBlock
        self.crossAttentionBlock = crossAttentionBlock
        self.feedForwardBlock = feedForwardBlock
        self.residualConnection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
        
    def forward(self, x, encoderOutput, srcMask, targetMask):
        # Apply self-attention, cross-attention, and feed-forward with residual connections
        x = self.residualConnection[0](x, lambda x: self.selfAttentionBlock(x, x, x, targetMask))
        x = self.residualConnection[1](x, lambda x: self.crossAttentionBlock(x, encoderOutput, encoderOutput, srcMask))
        x = self.residualConnection[2](x, self.feedForwardBlock)
        return x

# Decoder consisting of multiple decoder blocks
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, encoderOutput, srcMask, targetMask):
        # Pass through each decoder block
        for layer in self.layers:
            x = layer(x, encoderOutput, srcMask, targetMask)
        return self.norm(x)

# Projection layer to map model outputs to vocabulary size
class ProjectionLayer(nn.Module):
    def __init__(self, dimModel, vocabSize) -> None:
        super().__init__()
        self.projection = nn.Linear(dimModel, vocabSize)
        
    def forward(self, x) -> None:
        return self.projection(x)
        
# Transformer model with encoder, decoder, and embedding layers
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, srcEmbed: InputEmbedding, targetEmbed: InputEmbedding, srcPOS: PositionalEmbedding, targetPOS: PositionalEmbedding, projectionLayer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.srcEmbed = srcEmbed
        self.targetEmbed = targetEmbed
        self.srcPOS = srcPOS
        self.targetPOS = targetPOS
        self.projectLayer = projectionLayer
    
    def encode(self, src, srcMask):
        # Embed and encode source sequence
        src = self.srcEmbed(src)
        src = self.srcPOS(src)
        return self.encoder(src, srcMask)
        
    def decode(self, encoderOutput: torch.Tensor, srcMask: torch.Tensor, target: torch.Tensor, targetMask: torch.Tensor):
        # Embed and decode target sequence
        target = self.targetEmbed(target)
        target = self.targetPOS(target)
        return self.decoder(target, encoderOutput, srcMask, targetMask)
        
    def project(self, x):
        # Project model output to vocabulary space
        return self.projectLayer(x)
        
# Function to build the entire Transformer model
def buildTransformer(srcVocabSize: int, targetVocabSize: int, srcSeqLen: int, targetSeqLen: int, dimModel: int = 512, numLayers: int = 6, numHeads: int = 8, dropout: float = 0.1, feedForwardDim: int = 2048) -> Transformer:
    # Create input embeddings
    srcEmbed = InputEmbedding(dimModel, srcVocabSize)
    targetEmbed = InputEmbedding(dimModel, targetVocabSize)
    
    # Create positional embeddings
    srcPOS = PositionalEmbedding(dimModel, srcSeqLen, dropout)
    targetPOS = PositionalEmbedding(dimModel, targetSeqLen, dropout)
    
    # Build encoder blocks
    encoderBlocks = []
    for _ in range(numLayers):
        encoderSelfAttentionBlock = MultiheadAttentionBlock(dimModel, numHeads, dropout)
        feedForwardBlock = FeedForwardBlock(dimModel, feedForwardDim, dropout)
        encoderBlock = EncoderBlock(dimModel, encoderSelfAttentionBlock, feedForwardBlock, dropout)
        encoderBlocks.append(encoderBlock)
    
    # Build decoder blocks
    decoderBlocks = []
    for _ in range(numLayers):
        decoderSelfAttentionBlock = MultiheadAttentionBlock(dimModel, numHeads, dropout)
        decoderCrossAttentionBlock = MultiheadAttentionBlock(dimModel, numHeads, dropout)
        feedForwardBlock = FeedForwardBlock(dimModel, feedForwardDim, dropout)
        decoderBlock = DecoderBlock(dimModel, decoderSelfAttentionBlock, decoderCrossAttentionBlock, feedForwardBlock, dropout)
        decoderBlocks.append(decoderBlock)
    
    # Create encoder and decoder
    encoder = Encoder(dimModel, nn.ModuleList(encoderBlocks))
    decoder = Decoder(dimModel, nn.ModuleList(decoderBlocks))
    
    # Create projection layer
    projectionLayer = ProjectionLayer(dimModel, targetVocabSize)
    
    # Create Transformer model
    transformer = Transformer(encoder, decoder, srcEmbed, targetEmbed, srcPOS, targetPOS, projectionLayer)

    # Initialize model parameters with Xavier uniform distribution
    for parameter in transformer.parameters():
        if parameter.dim() > 1: 
            nn.init.xavier_uniform_(parameter)

    return transformer
