"""
9.1 门控循环神经网络
"""

from dataclasses import dataclass

import torch
from torch import nn as NN
from d2l import torch as d2l

from utils.data.time_machine import load_data_time_machine


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> substitute Tensor.__repr__
_ = __import__("sys")
if "torch" in _.modules:
    _ = __import__("torch")
    if not hasattr(_.Tensor, "_custom_repr_installed"):
        original_tensor_repr = _.Tensor.__repr__

        def custom_tensor_repr(self):
            return f"{tuple(self.shape)}{original_tensor_repr(self)}"

        setattr(_.Tensor, "__repr__", custom_tensor_repr)
        setattr(_.Tensor, "_custom_repr_installed", True)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< substitute Tensor.__repr__


@dataclass
class Config:
    batch_size: int = 32
    num_steps: int = 35
    num_epochs: int = 500
    lr: float = 1
    embedding_size: int = 256

    def __post_init__(self):
        _, vocab = load_data_time_machine(self.batch_size, self.num_steps)
        self.vocab_size = len(vocab)
        self.device = d2l.try_gpu()


def get_params(vocab_size, embedding_size, device):
    """
    初始化模型的所有参数
    """
    input_size, output_size = vocab_size, vocab_size

    def normal(shape, device):
        """给定shape, 返回一个正态分布的tensor. 0.01是缩放因子"""
        return torch.randn(shape, device=device) * 0.01

    def three():
        """
        返回3个tensor, 分别是(vocab_size, embedding_size), (vocab_size, embedding_size), (embedding_size)
        """
        return (
            normal((vocab_size, embedding_size), device=device),
            normal((embedding_size, embedding_size), device=device),
            torch.zeros(embedding_size, device=d2l.try_gpu()),
        )

    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()

    # W_hq: 隐藏层 -> 输出层的映射, shape = (, vocab_size)
    W_hq = normal((embedding_size, output_size), device=device)
    # b_q: 输出层的偏置, shape = (vocab_size)
    b_q = torch.zeros(output_size, device=device)

    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params


def init_GRU_hidden_state(batch_size, embedding_size, device):
    """初始化GRU的隐藏状态"""
    return (torch.zeros((batch_size, embedding_size), device=device),)


def GRU_forward(inputs, state, params):
    """
    GRU的前向传播

    Args:
        inputs: 输入, shape = (seq_length, batch_size, vocab_size)
        state: 上一个隐藏状态, shape = (batch_size, embedding_size)

    Returns:
        outputs: 当前输入所有时间步的hidden_state, shape = (seq_length * batch_size, embedding_size)
        (H, ): 当前输入最后一个时间步的hidden state, shape = (batch_size, embedding_size)
    """
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    (H,) = state
    outputs = []

    # 遍历seq_length
    for X in inputs:
        # update gate, shape = (batch_size, embedding_size)
        Z = torch.sigmoid(X @ W_xz + H @ W_hz + b_z)
        # reset gate, shape = (batch_size, embedding_size)
        R = torch.sigmoid(X @ W_xr + H @ W_hr + b_r)
        # candidate hidden state, shape = (batch_size, embedding_size)
        H_tilde = torch.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
        # true hidden state, shape = (batch_size, embedding_size)
        H = (1 - Z) * H_tilde + Z * H
        # output, shape = (batch_size, vocab_size)
        Y = H @ W_hq + b_q

        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)


if __name__ == "__main__":
    config = Config()

    # 从零实现
    train_iter, vocab = load_data_time_machine(
        batch_size=Config.batch_size, num_steps=Config.num_steps
    )
    model = d2l.RNNModelScratch(
        config.vocab_size,
        config.embedding_size,
        config.device,
        get_params=get_params,
        init_state=init_GRU_hidden_state,
        forward_fn=GRU_forward,
    )
    d2l.train_ch8(model, train_iter, vocab, config.lr, config.num_epochs, config.device)

    # 简介实现
    gru_layer = NN.GRU(config.vocab_size, config.embedding_size)
    model = d2l.RNNModel(gru_layer, config.vocab_size)
    model = model.to(config.device)
    d2l.train_ch8(
        model,
        train_iter=train_iter,
        vocab=vocab,
        lr=config.lr,
        num_epochs=config.num_epochs,
        device=config.device,
    )
