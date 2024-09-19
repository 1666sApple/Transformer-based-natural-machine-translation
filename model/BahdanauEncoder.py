import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttentionQKV(nn.Module):
    def __init__(self, hidden_size, query_size=None, key_size=None, dropout_p=0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_size = hidden_size if query_size is None else query_size
        self.key_size = 2 * hidden_size if key_size is None else key_size
        
        # Define linear layers for query, key, and energy calculation
        self.query_layer = nn.Linear(self.query_size, hidden_size)
        self.key_layer = nn.Linear(self.key_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)
        
        # Define dropout layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, hidden, encoder_outputs, src_mask=None):
        # hidden: (B, H) - the hidden state of the decoder
        # encoder_outputs: (Src_len, B, 2*H) - outputs from the encoder
        
        # Apply linear transformation to hidden state (query)
        query_out = self.query_layer(hidden)  # (B, H)
        
        # Apply linear transformation to encoder outputs (keys)
        key_out = self.key_layer(encoder_outputs)  # (Src_len, B, H)
        
        # We add the query to the key; broadcast query_out to (Src_len, B, H)
        energy_input = torch.tanh(query_out.unsqueeze(0) + key_out)  # (Src_len, B, H)
        
        # Calculate energy scores: (Src_len, B, H) -> (Src_len, B, 1) -> (Src_len, B)
        energies = self.energy_layer(energy_input).squeeze(2)  # (Src_len, B)
        
        # Apply mask (if provided) to exclude padding tokens from attention scores
        if src_mask is not None:
            energies = energies.masked_fill(src_mask == 0, float('-inf'))
        
        # Apply softmax over the source length dimension to get attention weights
        weights = F.softmax(energies, dim=0)  # (Src_len, B)
        
        # Return attention weights in (B, Src_len) format as expected
        return weights.transpose(0, 1)  # (B, Src_len)
