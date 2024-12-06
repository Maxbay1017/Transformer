import token
import torch 
import torch.nn as nn
from pathlib import Path

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer,models, trainers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataclasses import dataclass

@dataclass
class Config:
    datasource : str = 'opus_books'
    lang_src : str = 'en'
    lang_tgt : str = 'it'   
    tokenizer_file: str = 'tokenizer_{0}.json'
    unk_token: str = "[UNK]"
    special_tokens: list = ("[UNK]", "[PAD]", "[SOS]", "[EOS]")
    min_frequency: int = 2

@dataclass
class BPEConfig:
    datasource : str = 'opus_books'
    lang_src : str = 'en'
    lang_tgt : str = 'zh'
    tokenizer_file: str = 'BPEtokenizer_{0}.json'
    vocab_size: int = 30000  # 词汇表大小
    min_frequency: int = 2   # 子词的最小频率
    special_tokens: list = ("[UNK]", "[PAD]", "[SOS]", "[EOS]")


def get_all_sentences(ds:Dataset,lang:str):
    for item in ds:
        yield item['translation'][lang]

def get_or_train_tokenizer(config,ds:Dataset,lang:str):
    tokenizer_path = Path(config.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2) #在训练数据中出现频率大于或等于2次的词汇才会被包含在词汇表中，出现频率小于2次的词汇将被忽略，通常会被替换成未知词（[UNK]）标记。
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_or_train_BPETokenizer(config:dataclass,lang:str):
    tokenizer_path = Path(config.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        # 初始化 BPE Tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))    
    else:
        pass
    return tokenizer 