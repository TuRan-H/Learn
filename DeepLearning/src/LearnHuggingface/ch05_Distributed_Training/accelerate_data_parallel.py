"""
使用accelerate框架进行并行训练
"""

import os
import torch as tc
import numpy as np
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from src.LearnHuggingface.ch05_Distributed_Training.arguments import Arguments


LOCAL_RANK = os.environ["LOCAL_RANK"]


class MyDataset(Dataset):
    def __init__(self, args: Arguments) -> None:
        super().__init__()
        self.data = pd.read_csv(args.dataset_path)
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


def prepare_dataloader(args: Arguments):
    dataset = MyDataset(args)

    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=tc.Generator().manual_seed(42))

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = tc.tensor(labels)
        return inputs

    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader



def prepare_model_and_optimizer(args: Arguments):

    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path)

    optimizer = Adam(model.parameters(), lr=args.lr)

    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with tc.inference_mode():
        for batch in validloader:
            output = model(**batch)
            pred = tc.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)


def train(model, optimizer, train_loader, valid_loader, accelerator: Accelerator, args: Arguments):
    global_step = 0
    logs_loss = np.array(0)
    bar = None

    if LOCAL_RANK == 0:
        bar = tqdm(total=args.num_epoch * len(train_loader))

    for ep in range(args.num_epoch):
        model.train()
        for batch in train_loader:
            output = model(**batch)
            loss = output.loss
            accelerator.backward(loss)
            optimizer.step()
            if global_step % args.log_step == 0:
                logs_loss = loss.detach().cpu().numpy()

            if LOCAL_RANK == 0:
                assert isinstance(bar, tqdm)
                bar.update(1)
                bar.set_postfix({"loss": logs_loss})

def main():
    args = Arguments()
    accelerator = Accelerator()
    train_dataloader, valid_dataloader = prepare_dataloader(args)
    model, optimizer = prepare_model_and_optimizer(args)
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader)

    train(model, optimizer, train_dataloader, valid_dataloader, accelerator, args)


if __name__ == "__main__":
    main()