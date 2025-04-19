"""
9.2 长短期记忆网络
"""

from dataclasses import dataclass, field

import torch
from torch.nn import functional as functional
from torch import nn as NN
from d2l import torch as d2l

from utils.data.time_machine import load_data_time_machine


@dataclass
class Config:
    batch_size: int = field(default=32, metadata={"help": "batch size"})
    seq_length: int = field(default=35, metadata={"help": "sequence length 或者 num steps"})
    embedding_size: int = field(default=256, metadata={"help": "embedding size 或者 num_hiddens"})
    vocab_size: int = field(default=28, metadata={"help": "vocab size"})
    num_epochs: int = field(default=100, metadata={"help": "总的epoch数"})
    lr: float = field(default=1, metadata={"help": "学习率"})
    device: torch.device = d2l.try_gpu()


def get_LSTM_params(vocab_size, embedding_size, device):
    """
    获取LSTM中的所有参数
    """

    def normal(shape):
        return torch.randn(shape, device=device) * 0.01

    def three(vocab_size, embedding_size):
        return (
            normal((vocab_size, embedding_size)),
            normal((embedding_size, embedding_size)),
            normal((embedding_size)),
        )

    W_xi, W_hi, b_i = three(vocab_size, embedding_size)  # 输入门
    W_xf, W_hf, b_f = three(vocab_size, embedding_size)  # 遗忘门
    W_xo, W_ho, b_o = three(vocab_size, embedding_size)  # 输出门
    W_xc, W_hc, b_c = three(vocab_size, embedding_size)  # 候选记忆单元

    W_hq = normal((embedding_size, vocab_size))
    b_q = normal((vocab_size))
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]

    for param in params:
        param.requires_grad = True

    return params


def init_LSTM_state(batch_size, embedding_size, device):
    """
    初始化LSTM的隐状态和记忆单元
    """
    return (
        torch.zeros((batch_size, embedding_size), device=device),
        torch.zeros((batch_size, embedding_size), device=device),
    )


def LSTM_forward(inputs, state, params):
    """
    LSTM前向传播

    Args:
        inputs: 输入数据, shape = (seq_length, batch_size, vocab_size)
        state: 隐状态和记忆单元 shape 均为 (batch_size, embedding_size)
    """
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    # 上一个时间步的隐状态和记忆单元
    H, C = state

    outputs = []
    for X in inputs:
        I = functional.sigmoid(X @ W_xi + H @ W_hi + b_i)
        F = functional.sigmoid(X @ W_xf + H @ W_hf + b_f)
        O = functional.sigmoid(X @ W_xo + H @ W_ho + b_o)
        C_tilde = functional.tanh(X @ W_xc + H @ W_hc + b_c)
        C = F * C + I * C_tilde
        H = O * functional.tanh(C)
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H, C)


if __name__ == "__main__":
    config = Config()

    # 从零实现
    train_iter, vocab = load_data_time_machine(Config.batch_size, Config.seq_length)
    config.vocab_size = len(vocab)
    model = d2l.RNNModelScratch(
        vocab_size=config.vocab_size,
        num_hiddens=config.embedding_size,
        device=config.device,
        get_params=get_LSTM_params,
        init_state=init_LSTM_state,
        forward_fn=LSTM_forward,
    )
    d2l.train_ch8(model, train_iter, vocab, config.lr, config.num_epochs, config.device)

    # 简介实现
    lstm_layer = NN.LSTM(config.vocab_size, config.embedding_size)
    model = d2l.RNNModel(lstm_layer, config.vocab_size)
    model = model.to(config.device)
    d2l.train_ch8(model, train_iter, vocab, config.lr, config.num_epochs, config.device)
