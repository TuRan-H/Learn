"""
动手实现一个transformer模型, 用于文本生成任务

参考资料:
* https://www.bilibili.com/video/BV1qWwke5E3K/?spm_id_from=333.1387.homepage.video_card.click&vd_source=41721633578b9591ada330add5535721
"""

import os
import torch
import tiktoken

from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from tqdm import tqdm

from src.transformer.modules.positional_embedding import PositionalEmbedding
from src.transformer.modules.group_query_attention import GroupQueryAttention
from src.transformer.models.transformer import Transformer
from src.transformer.datasets.seq_monkey import get_dataloader_seq_monkey


@dataclass
class Config:
    """
    创建配置参数类
    """

    # 训练参数
    seed: int = field(default=1024, metadata={"help": "随机数种子"})
    block_size: int = field(
        default=512,
        metadata={"help": "DataSet会将长序列切分为多个block, 每个block的长度为block_size. 等价于seq_len"},
    )
    batch_size: int = field(default=12, metadata={"help": "batch size"})
    vocab_size: int = field(default=50257, metadata={"help": "vocab size"})
    dropout_rate: float = field(default=0.1, metadata={"help": "dropout rate"})
    n_layer: int = field(default=6, metadata={"help": "decoder layer的层数"})
    n_head: int = field(default=12, metadata={"help": "MLA中head的数量"})
    embedding_dim: int = field(default=768, metadata={"help": "embedding dim"})
    head_dim: int = field(default=64, metadata={"help": "MLA中每个head的dim"})
    learning_rate: float = field(default=3e-4, metadata={"help": "learning rate"})
    T_max: int = field(default=1000, metadata={"help": "cosine annealing learning rate scheduler的超参"})
    device: str = field(default="cuda:0", metadata={"help": "device"})
    total_epoch: int = field(default=2, metadata={"help": "total epoch"})

    # 数据参数
    save_dir: str = field(default="./result/transformer", metadata={"help": "模型保存路径"})
    data_dir: str = field(
        default="./data/mobvoi_seq_monkey_general_open_corpus.jsonl",
        metadata={"help": "数据存放路径"},
    )

    # 其他参数
    is_train: bool = field(default=True, metadata={"help": "是否对模型进行训练"})
    is_eval: bool = field(default=False, metadata={"help": "是否对模型进行评估"})
    is_inference: bool = field(default=False, metadata={"help": "是否对模型进行推理"})
    use_cache_model: bool = field(default=False, metadata={"help": "推理模型的时候是否使用缓存的模型"})
    max_line: int = field(default=2000, metadata={"help": "读取数据集是所用的最大行数"})

    def __post_init__(self):
        assert self.embedding_dim % self.n_head == 0, "n_embd must be divided by n_head"
        self.head_dim = self.embedding_dim // self.n_head
        torch.manual_seed(self.seed)


def train_epoch(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    device: str,
    epoch: int,
):
    # 首先将model设置为训练模式, 开启dropout
    model.train()
    total_loss = 0.0
    previous_steps = epoch * len(train_loader)

    bar = tqdm(total=len(train_loader))
    for i, batch in enumerate(train_loader):
        # ! 将模型的梯度置零, 防止梯度累加. (tips: 只需要在loss.backward()之前置零即可)
        optimizer.zero_grad()
        x, y = batch['input_ids'].to(device), batch['labels'].to(device)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        bar.update(1)
        bar.set_description(f"Train Epoch: {epoch}")
    scheduler.step()

    return total_loss


def validate(model: Transformer, val_dataloader: DataLoader, device: str, epoch: int):
    model.eval()
    total_loss = 0.0

    bar = tqdm(total=len(val_dataloader))
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            x, y = batch['input_ids'].to(device), batch['labels'].to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            bar.update(1)
            bar.set_description(f"Validate Epoch: {epoch}")

    return total_loss


def evaluate(
    config: Config,
    model: Transformer,
    dev_dataloader: DataLoader,
):
    """
    评估模型
    """
    model.eval()
    device = config.device

    bar = tqdm(total=len(dev_dataloader))
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            x, y = batch['input_ids'].to(device), batch['labels'].to(device)
            logits, loss = model(x, y)
            bar.update(1)
            bar.set_description(f"Evaluate")


def train(
    config: Config,
    model: Transformer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """
    训练和保存模型
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max)
    model = model.to(config.device)
    for epoch in range(config.total_epoch):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_dataloader,
            device=config.device,
            epoch=epoch,
        )
        val_loss = validate(
            model=model,
            val_dataloader=val_dataloader,
            device=config.device,
            epoch=epoch,
        )

    # 保存模型
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    save_path = os.path.join(config.save_dir, 'transformer.pth')
    torch.save(checkpoint, save_path)


def inference(
    config: Config,
    model: Transformer,
):
    """
    使用模型进行推理
    """
    if config.use_cache_model:
        checkpoint_path = os.path.join(config.save_dir, "transformer.pth")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    tokenizer = tiktoken.get_encoding('gpt2')
    input_ids = tokenizer.encode("who are you")
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    input_ids = input_ids.to(config.device)
    outputs = model.generate(input_ids, 100)
    outputs = outputs.tolist()
    outputs = tokenizer.decode(outputs[0])
    print(f"Generated text: {outputs}")


if __name__ == '__main__':
    # *** 配置参数
    config = Config(use_cache_model=False, is_train=True, is_eval=True, is_inference=False)

    # *** 导入数据集
    train_dataloader, val_dataloader, dev_dataloader = get_dataloader_seq_monkey(
        path=config.data_dir, block_size=config.block_size, batch_size=config.batch_size, shuffle=True
    )

    # *** 创建模型
    positional_embedding = PositionalEmbedding(seq_len=config.block_size, embedding_dim=config.embedding_dim)
    group_query_attention = GroupQueryAttention(
        hidden_dim=config.embedding_dim,
        nums_head=config.n_head,
        nums_kv_head=int(config.n_head / 2),
        attention_dropout_rate=config.dropout_rate,
    )

    model = Transformer(
        embedding_dim=config.embedding_dim,
        dropout_rate=config.dropout_rate,
        vocab_size=config.vocab_size,
        max_seq_length=config.block_size,
        nums_layer=config.n_layer,
        positional_encoding=positional_embedding,
        attention=group_query_attention,
    )

    if config.is_train:
        train(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

    if config.is_eval:
        evaluate(config=config, model=model, dev_dataloader=dev_dataloader)

    if config.is_inference:
        inference(config=config, model=model)
