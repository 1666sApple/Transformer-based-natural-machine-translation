import torch
import torch.nn as nn
from torch.utils.data import Dataset

def causalMask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    def __init__(self, dataset, srcTokenizer, targetTokenizer, srcLang, targetLang, seqLen):
        super().__init__()
        self.seqLen = seqLen
        self.dataset = dataset
        self.srcTokenizer = srcTokenizer
        self.targetTokenizer = targetTokenizer
        self.srcLang = srcLang
        self.targetLang = targetLang
        
        # Start of sentence
        self.SOS = torch.tensor([targetTokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        # End of sentence
        self.EOS = torch.tensor([targetTokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        # Padding
        self.PAD = torch.tensor([targetTokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        srcTargetPair = self.dataset[idx]
        srcText = srcTargetPair['translation'][self.srcLang]
        targetText = srcTargetPair['translation'][self.targetLang]
        
        encoderInputTokens = self.srcTokenizer.encode(srcText).ids
        decoderInputTokens = self.targetTokenizer.encode(targetText).ids  
        
        srcPadCount = self.seqLen - len(encoderInputTokens) - 2
        targetPadCount = self.seqLen - len(decoderInputTokens) - 1  
        
        if srcPadCount < 0 or targetPadCount < 0:
            raise ValueError('Sentence too long!')
        
        encoderInput = torch.cat(
            [
                self.SOS, 
                torch.tensor(encoderInputTokens, dtype=torch.int64),
                self.EOS,
                torch.tensor([self.PAD] * srcPadCount, dtype=torch.int64)
            ], dim=0
        )
        
        decoderInput = torch.cat(
            [
                self.SOS, 
                torch.tensor(decoderInputTokens, dtype=torch.int64),
                torch.tensor([self.PAD] * targetPadCount, dtype=torch.int64)
            ], dim=0
        )
        
        labels = torch.cat(
            [
                torch.tensor(decoderInputTokens, dtype=torch.int64),
                self.EOS,
                torch.tensor([self.PAD] * targetPadCount, dtype=torch.int64)
            ], dim=0
        )
        
        assert encoderInput.size(0) == self.seqLen
        assert decoderInput.size(0) == self.seqLen
        assert labels.size(0) == self.seqLen
        
        return {
            "encoderInput": encoderInput,
            "decoderInput": decoderInput,
            "encoderMask": (encoderInput != self.PAD).unsqueeze(0).unsqueeze(0).int(),
            "decoderMask": (decoderInput != self.PAD).unsqueeze(0).int() & causalMask(decoderInput.size(0)),
            "label": labels,
            "srcText": srcText,
            "targetText": targetText
        }