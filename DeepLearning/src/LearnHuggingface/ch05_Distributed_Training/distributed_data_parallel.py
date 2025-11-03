"""
b站教学视频: 【手把手带你实战HuggingFace Transformers-分布式训练篇】分布式数据并行原理与应用

代码流程:
    1. 使用pytorch框架对预训练的语言模型进行微调, 实现文本的情感分类任务
    2. 使用Distributed data parallel进行分布式训练

backbone: https://huggingface.co/hfl/rbt3
corpus: https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv

注意: 本篇代码需要使用 `torchrun` 来运行
"""

import os
import pandas as pd
import torch
import torch.distributed as distributed
from typing import cast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    AutoModelForSequenceClassification,
    HfArgumentParser,
)
from src.LearnHuggingface.ch05_Distributed_Training.arguments import Arguments


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")


class MyDataset(Dataset):
    def __init__(self, dataset_path: str | None = None) -> None:
        super().__init__()
        assert isinstance(dataset_path, str), "Please provide a valid dataset path."
        self.data = pd.read_csv(dataset_path)
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]['review'], self.data.iloc[index]['label']

    def __len__(self):
        return len(self.data)


class MyDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch):
        tokenizer = self.tokenizer
        instances, labels = zip(*batch)
        inputs = tokenizer(instances, max_length=128, padding='max_length', truncation=True, return_tensors='pt')  # type: ignore
        inputs['labels'] = torch.tensor(labels)

        return inputs


def load_model(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

    return tokenizer, model


def train(
    model: PreTrainedModel,
    optimizer: torch.optim.Adam,
    dataloader: DataLoader,
    **kwargs,
):
    """
    训练函数，用于训练文本分类模型。
    """
    # 将model放到GPU上
    epoch = kwargs.pop('epoch')
    global_step = 0

    # 将模型调整至训练模式
    model.train()
    for ep in range(epoch):
        # ATTN: set_epoch()方法能够修改sampler的随机数索引顺序, 使得每个epoch中数据划分不同, 增强训练鲁棒性
        dataloader.sampler.set_epoch(ep)  # type: ignore
        # ATTN: 仅在local rank 0上显示进度条
        bar = None
        if LOCAL_RANK == 0:
            bar = tqdm(total=len(dataloader), desc=f"rank = {LOCAL_RANK}, epoch = {ep+1}")
        for input in dataloader:
            input = {k: v.to(DEVICE) for k, v in input.items()}
            output = model(**input)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            # 在每一个batch的梯度更新结束后, 置零梯度, 防止梯度累计
            optimizer.zero_grad()
            global_step += 1

            if LOCAL_RANK == 0:
                assert isinstance(bar, tqdm)
                bar.update(1)
                bar.set_postfix({"loss": loss.detach().item()})


def evaluate(model: PreTrainedModel, dataloader: DataLoader):
    """
    Evaluate the performance of a text classification model on a given dataset.
    """
    acc_num = torch.tensor(0.0, device=DEVICE)
    total_count = torch.tensor(0.0, device=DEVICE)
    with torch.inference_mode():
        for input in tqdm(dataloader):
            input = {k: v.to(DEVICE) for k, v in input.items()}
            output = model(**input)
            # shape output['logits'] (batch_size, num_classes)
            predictions = torch.argmax(output['logits'], dim=-1)
            acc_num += (predictions.long() == input['labels']).long().float().sum()
            total_count += input['labels'].size(0)

    # ATTN: 因为数据集被分块给每一个进程了, 因此每个进程求解出来的acc_num只有一部分. 在这里要对不同进程的acc_num和total_count求和
    distributed.all_reduce(acc_num)
    distributed.all_reduce(total_count)

    accuracy = acc_num / total_count
    return accuracy.item()


def main():
    parser = HfArgumentParser((Arguments))  # type: ignore

    (args,) = parser.parse_args_into_dataclasses()
    args = cast(Arguments, args)
    distributed.init_process_group(backend="nccl")

    # 导入模型和分词器
    tokenizer, model = load_model(args.model_name_or_path)

    model.to(DEVICE)
    # ATTN: 将模型导入到某张GPU后, 再去包装模型
    model = DistributedDataParallel(model)

    # 导入优化器
    optimizer = Adam(model.parameters(), lr=2e-5)

    # 划分数据集
    all_dataset = MyDataset(args.dataset_path)
    # * 由于每个进程之间的数据划分不一样, 可能会导致数据泄露问题, 这里将每个进程按照一致的方式划分
    train_dataset, valid_dataset = random_split(all_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    # 实例化DataCollator
    datacollator = MyDataCollator(tokenizer)

    # * 修改DataLoader的sampler为DistributedSampler, 使得不同的进程能够获取
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=datacollator,
        sampler=DistributedSampler(train_dataset),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=64,
        collate_fn=datacollator,
        sampler=DistributedSampler(valid_dataset),
    )

    # 开始训练
    train(model, optimizer, train_dataloader, epoch=3)  # type: ignore

    # 开始评估, 计算评价指标
    metrics = evaluate(model, valid_dataloader)  # type: ignore
    if LOCAL_RANK == 0:
        print(metrics)

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
