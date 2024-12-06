from datasets import Dataset,load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFKC
from pathlib import Path

def get_all_sentences(ds:Dataset,lang:str):
    for item in ds:
        yield item['translation'][lang]

def train_tokenizer(
        input_ds: str | Dataset, #  The directory containing the dataset.
        lang:str, # language of the dataset
        save_path: str,
        tokenizer_type: str = "BPE",
        vocab_size: int = 52000,
):
    
    """
    
    Trains a tokenizer on all the sentences in the dataset.

    Args:
        input_ds: The directory containing the dataset.
        save_path: The path to save the tokenizer.
        tokenizer_type: The type of tokenizer to train. Default is "BPE".
        vocab_size: The size of the vocabulary. Default is 52000.

    Returns:
        The trained tokenizer.
    
    """

    if tokenizer_type == "BPE":
        model = models.BPE()
    else:
        raise ValueError(f"Tokenizer type {tokenizer_type} not supported.")
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(model)
    
    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True) # 预先将文本进行字节级别的切分 add_prefix_space=True表示在文本前添加一个空格 example: "你好世界" -> [(' ', '你'), (' ', '好'), (' ', '，'), (' ', '世'), (' ', '界'), (' ', '！')]
    tokenizer.decoder = decoders.ByteLevel() # 将字节级别的切分结果进行解码
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True) # 去除切分结果中的空格
    tokenizer.normalizer = NFKC() # 对文本进行NFKC规范化

    # And then train it
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>","<|padding|>"]
    )

    tokenizer.train_from_iterator(get_all_sentences(input_ds,lang),trainer=trainer)
    # tokenizer.train_from_iterator(input_ds, trainer=trainer)

    # Save the tokenizer
    if save_path:
        tokenizer.save(save_path,pretty=True) # 保存tokenizer prefix=True表示在保存时添加前缀
        print(f"Tokenizer saved to {save_path}")
    
    return tokenizer



def train_bbpe_tokenizer(input_ds, lang, vocab_size=52000, save_path="bbpe_tokenizer_{0}"):
    """
    使用 Byte-Level BPE 训练一个支持中英文的分词器。

    Args:
        input_ds: 数据集，包含文本句子的迭代器。
        lang: 数据集语言。
        vocab_size: 词汇表大小。
        save_path: 分词器保存路径。
    """
    # Step 1: Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Step 2: Customize pre-tokenization and decoding (Optional)
    # 归一化（NFKC 标准化，处理全角/半角字符等问题）
    tokenizer.normalizer = NFKC()

    # 字节级别的分词器
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 字节级别的解码器
    tokenizer.decoder = decoders.ByteLevel()

    # Step 3: Train the tokenizer using the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],  # 特殊 token
        special_tokens=["<|endoftext|>","<|padding|>"]
    )

    # 从数据集迭代器中训练分词器
    tokenizer.train_from_iterator(
        get_all_sentences(input_ds, lang), trainer=trainer
    )

    # Step 4: Save the tokenizer
    save_path = Path(save_path.format(lang))
    tokenizer.save(f"{save_path}.json", pretty=True)
    print(f"Tokenizer saved to {save_path}.json")

    return tokenizer
