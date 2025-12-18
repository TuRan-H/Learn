import torch
import json
import tiktoken

from torch.utils.data import Dataset
from tqdm import tqdm


class SeqMonkeyDataset(Dataset):
    def __init__(self, path: str, block_size: int) -> None:
        """
        Args:
            path: 数据集路径
            block_size: 将序列进行切分, 单个序列大小
        """
        self.block_size = block_size
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = self.tokenizer.encode(self.eos_token, allowed_special="all")[0]
        self.max_line = 2000

        # 读取所有原始数据
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_line:
                    break
                raw_data.append(json.loads(line)['text'])

        # 将所有原始数据编码, 并且拼接为一个长序列
        full_encoded = []
        for text in raw_data:
            encoded = self.tokenizer.encode(text)
            full_encoded.extend(encoded + [self.eos_token_id])

        # 将长序列切分为多个block, 作为训练数据
        self.encoded_data = []
        bar = tqdm(range(0, len(full_encoded), self.block_size), desc="Encoding data")
        for i in range(0, len(full_encoded), self.block_size):
            # 每个block的长度为block_size + 1, 前block_size作为输入, 最后一个token作为训练目标
            block = full_encoded[i : i + self.block_size + 1]
            # 如果最后一个block的大小不够, 使用eos_token_id作为padding
            if len(block) < self.block_size + 1:
                block += [self.eos_token_id] * (self.block_size + 1 - len(block))
            self.encoded_data.append(block)
            bar.update(1)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        block = self.encoded_data[idx]

        inputs = torch.tensor(block[:-1], dtype=torch.long)
        targets = torch.tensor(block[1:], dtype=torch.long)

        data = {"input_ids": inputs, "labels": targets}

        return data


def get_dataloader_seq_monkey(path: str, block_size: int, batch_size: int, shuffle: bool):
    dataset = SeqMonkeyDataset(path=path, block_size=block_size)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.5, 0.2, 0.3])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader
