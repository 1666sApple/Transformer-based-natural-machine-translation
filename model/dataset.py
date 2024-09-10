import torch
from torch.utils.data import Dataset
import numpy as np

def causalMask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    def __init__(self, dataset, srcTokenizer, targetTokenizer, srcLang, targetLang, seqLen):
        super().__init__()
        self.dataset = dataset
        self.srcTokenizer = srcTokenizer
        self.targetTokenizer = targetTokenizer
        self.srcLang = srcLang
        self.targetLang = targetLang
        self.seqLen = seqLen

        self.sos_token = torch.tensor([targetTokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([targetTokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([targetTokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair['translation'][self.srcLang]
        target_text = src_target_pair['translation'][self.targetLang]

        enc_input_tokens = self.srcTokenizer.encode(src_text).ids
        dec_input_tokens = self.targetTokenizer.encode(target_text).ids

        enc_num_padding_tokens = self.seqLen - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seqLen - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add SOS and EOS to encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add SOS to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add EOS to label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seqLen
        assert decoder_input.size(0) == self.seqLen
        assert label.size(0) == self.seqLen

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causalMask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "target_text": target_text,
        }