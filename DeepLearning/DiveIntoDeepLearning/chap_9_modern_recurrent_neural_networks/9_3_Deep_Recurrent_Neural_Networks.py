from dataclasses import dataclass

import torch
from torch import nn
from d2l import torch as d2l

from utils.data.time_machine import load_data_time_machine


@dataclass
class Config:
    batch_size: int = 32
    num_steps: int = 35
    vocab_szie: int = 28
    num_hiddens: int = 256
    num_layers: int = 2
    device: torch.device = d2l.try_gpu()
    num_epochs: int = 500
    lr: float = 1


if __name__ == "__main__":
    config = Config()
    train_iter, vocab = load_data_time_machine(config.batch_size, config.num_steps)
    lstm_layer = nn.LSTM(config.vocab_szie, config.num_hiddens, config.num_layers)
    model = d2l.RNNModel(lstm_layer, config.vocab_szie)
    model = model.to(config.device)
    d2l.train_ch8(model, train_iter, vocab, config.lr, config.num_epochs, config.device)
