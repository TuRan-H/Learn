"""8.5 循环神经网络的从零开始实现"""

import math
from typing import Any, Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
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
    batch_size: int = field(default=32, metadata={"help": "batch size"})
    num_steps: int = field(default=35, metadata={"help": "总的steps的大小"})
    num_hiddens: int = field(default=512, metadata={"help": "隐藏层的大小"})
    num_epochs: int = field(default=500, metadata={"help": "总的epoch"})
    lr: float = field(default=1, metadata={"help": "学习率"})


def get_params(vocab_size, num_hiddens, device):
    """
    Initialize the model parameters.

    Args:
        vocab_size (int): The size of the vocabulary.
        num_hiddens (int): The number of hidden units.
        device (torch.device): The device to store the parameters.

    Returns:
        tuple: A tuple containing the initialized parameters.
    """
    num_inputs = vocab_size  # num_inputs: 模型输入的维度
    num_outputs = vocab_size  # num_outputs: 模型输出的维度

    W_xh = torch.randn((num_inputs, num_hiddens), device=device) * 0.01
    W_hh = torch.randn((num_hiddens, num_hiddens), device=device) * 0.01
    b_h = torch.zeros(num_hiddens, device=device)

    W_hq = torch.randn((num_hiddens, num_outputs), device=device) * 0.01
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params


def init_rnn_state(batch_size, num_hiddens, device):
    """初始化RNN隐藏状态"""
    return torch.zeros(batch_size, num_hiddens, device=device)


def rnn(inputs, state, params):
    """
    RNN的前向传播

    Args:
        inputs (torch.Tensor): The input data, shape (num_steps, batch_size, vocab_size).
        state (torch.Tensor): The initial hidden state, shape (batch_size, num_hiddens).
        params (list): The list of model parameters.

    Returns:
        tuple: A tuple containing the outputs and the final hidden state.
                outputs shape: (num_steps * batch_size, vocab_size).
                final hidden state shape: (batch_size, num_hiddens).
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []

    for X in inputs:
        # H: `X_tW_{xh} + H_{t-1}W_{hh}`, 生成当时间步的隐藏状态
        # H shape: (batch_size, num_hiddens)
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        # Y shape: (batch_size, vocab_size)
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), H


# *** 定义RNN类
class RNNModelScartch:
    def __init__(
        self,
        vocab_size,
        num_hiddens,
        device,
        get_params: Callable,
        init_state: Callable,
        forward_fn: Callable,
    ) -> None:
        """
        初始化RNN模型

        Args:
            vocab_size (int): The size of the vocabulary.
            num_hiddens (int): The number of hidden units.
            device (torch.device): The device to store the parameters.
            get_params (Callable): 默认是 `get_params` 函数.
            init_state (Callable): 默认是 `init_rnn_state` 函数
            forward_fn (Callable): 默认是 `rnn` 函数
        """
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(self.vocab_size, self.num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, X, state) -> Any:
        """
        RNN类的Forward函数

        Args:
            X (Tensor): Input tensor of shape (batch_size, num_steps).
                `num_steps` 表示时间步, 也就是输入序列的长度
            state (Any): The initial state of the RNN

        Returns:
            Any: The output of the forward function
        """

        # X shape: (batch_size, num_steps) -> (num_steps, batch_size, vocab_size)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """
        Initialize the hidden state of the RNN model.

        Args:
            batch_size (int): The size of the batch.
            device (torch.device): The device to store the hidden state.

        Returns:
            torch.Tensor: The initialized hidden state, shape (batch_size, num_hiddens).
        """
        return self.init_state(batch_size, self.num_hiddens, device)


def predict(prefix: str, num_preds: int, net: RNNModelScartch, vocab, device):
    """
    使用RNN进行预测
    """
    # 获取初始状态
    state = net.begin_state(batch_size=1, device=device)
    # 将第一个token存到输出中
    outputs = [vocab[prefix[0]]]
    # 获取outputs中的最后一个token
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    # 预热
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 开始预测
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=-1).reshape(1)))

    return "".join(vocab.idx_to_token[i] for i in outputs)


def grad_clipping(net, theta):  # @save
    """
    裁剪梯度

    $\mathcal{g}  \gets \min (1, \frac{\theta}{||\mathcal{g} ||} )\mathcal{g}$
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))  # type: ignore
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm  # type: ignore


# *** 训练函数
def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """
    训练网络一个迭代周期（定义见第8章）

    Returns:
        tuple: (当前epoch的困惑度, 每个step的平均速度)
    """
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state   TODO indent up 1
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            state.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        l.backward()
        # 对神经网络参数进行裁剪
        grad_clipping(net, 1)
        # 因为已经调用了mean函数
        updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    # 初始化 optimizer
    optimizer = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict_function = lambda prefix: predict(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(net, train_iter, loss, optimizer, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict_function('time traveller'))

    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')  # type: ignore
    print(predict_function('time traveller'))
    print(predict_function('traveller'))


if __name__ == "__main__":
    config = Config()

    # 导入数据集
    train_iter, vocab = load_data_time_machine(config.batch_size, config.num_steps)

    # 实例化模型
    net = RNNModelScartch(
        vocab_size=len(vocab),
        num_hiddens=config.num_hiddens,
        device=d2l.try_gpu(),
        get_params=get_params,
        init_state=init_rnn_state,
        forward_fn=rnn,
    )

    # 训练
    train(
        net,
        train_iter=train_iter,
        vocab=vocab,
        lr=config.lr,
        num_epochs=config.num_epochs,
        device=d2l.try_gpu(),
    )
