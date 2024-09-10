import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def getAllSentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def getOrBuildTokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizerFile'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(getAllSentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def causalMask(size):
    # Define the causal mask function
    return torch.triu(torch.ones(size, size), diagonal=1).bool()

def greedyDecode(model, src, src_mask, src_tokenizer, target_tokenizer, maxLen, device):
    idx_sos = target_tokenizer.token_to_id('[SOS]')
    idx_eos = target_tokenizer.token_to_id('[EOS]')
    
    encoder_output = model.encode(src, src_mask)
    decoder_input = torch.empty(1, 1).fill_(idx_sos).type_as(src).to(device)
    
    while True:
        if decoder_input.size(1) == maxLen:
            break
    
        decoder_mask = causalMask(decoder_input.size(1)).type_as(src_mask).to(device)
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
        
        prob = model.project(out[:, -1])
        _, nxt_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src).fill_(nxt_word.item()).to(device)],
            dim=1
        )
        
        if nxt_word == idx_eos:
            break
        
    return decoder_input.squeeze(0)