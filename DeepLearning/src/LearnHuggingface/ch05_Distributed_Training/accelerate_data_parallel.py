"""
使用accelerate框架进行并行训练
"""

import math
import torch as tc
import numpy as np
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from src.LearnHuggingface.ch05_Distributed_Training.arguments import Arguments


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

    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch_size, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader


def prepare_model_and_optimizer(args: Arguments):

    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path)

    optimizer = Adam(model.parameters(), lr=args.lr)

    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with tc.no_grad():
        for batch in validloader:
            output = model(**batch)
            pred = tc.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)


def train(model, optimizer, train_loader, valid_loader, accelerator: Accelerator, args: Arguments):
    global_step = 0
    log_loss = np.array(0)
    bar = None

    if accelerator.is_main_process:
        bar = tqdm(total=math.ceil(args.num_epoch * len(train_loader) / args.gradient_accumulation_steps))

    for _ in range(args.num_epoch):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                # ATTN: 使用accelerator包装模型进行分布式训练后, 不要再使用loss.backward()了, 因为这样无法做到梯度的all-reduce. 使用accelerator.backward()代替
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    # ATTN: 分布式条件下, 使用reduce来获取所有进程的 mean loss
                    log_loss = accelerator.reduce(loss, reduction="mean")
                    log_loss = log_loss.item()  # type: ignore

                    if global_step % args.save_steps == 0:
                        accelerator.save_state(f"{args.output_dir}/step_{global_step}")
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=f"{args.output_dir}/step_{global_step}",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_func=accelerator.save,
                        )  # type: ignore

                    if accelerator.is_main_process:
                        accelerator.log(
                            {"train/loss": log_loss, "global_step": global_step},
                            step=global_step,
                        )

                        assert isinstance(bar, tqdm)
                        bar.update(1)
                        bar.set_postfix({"loss": log_loss})

        acc = evaluate(model, valid_loader, accelerator)
        accelerator.log(
            {"eval/accuracy": acc, "global_step": global_step},
            step=global_step,
        )

    if accelerator.is_main_process:
        assert isinstance(bar, tqdm)
        bar.close()


def main():
    args = Arguments()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
    )

    accelerator.init_trackers(
        project_name=args.project_name,
        init_kwargs={
            "wandb": {
                "name": args.run_name,
                "dir": args.log_dir,
            }
        },
    )

    train_dataloader, valid_dataloader = prepare_dataloader(args)
    model, optimizer = prepare_model_and_optimizer(args)
    # ATTN: 使用accelerator.prepare包装 model, optimizer, dataloader, scheduler (可选)
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
    )

    train(model, optimizer, train_dataloader, valid_dataloader, accelerator, args)

    # ATTN: 结束分布式训练, 这行代码很关键, 防止显存泄漏
    accelerator.end_training()


if __name__ == "__main__":
    main()
