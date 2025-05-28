import math

import torch
from torch import nn as NN
from torch.nn import functional as F


class LearnablePositionalEmbedding(NN.Module):
    """
    可学习的位置编码
    """

    def __init__(self, seq_len: int, embedding_dim: int):
        super().__init__()
        self.embedding = NN.Embedding(seq_len, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"
        Args:
            x (torch.Tensor): 输入tensor过word embedding, 形状为 (batch_size, seq_len, embedding_dim)
        """
        positional_embedding = self.embedding(torch.arange(x.size(1), device=x.device))
        
        return x + positional_embedding


class PositionalEmbedding(NN.Module):
    def __init__(self, seq_len: int, embedding_dim: int, scale_factor: float = 0.1):
        """
        transformer中的位置编码

        $$$
        PE(pos, 2i) = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)

        PE(pos, 2i+1) = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)
        $$$

        Args:
            seq_len: int, 序列长度
            embedding_dim: 嵌入维度的大小
        ):
        """
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim必须是偶数"
        pe = torch.zeros(seq_len, embedding_dim)
        # 分子部分
        numerator = torch.arange(seq_len, dtype=torch.float).unsqueeze(
            1
        )  # numerator -> (seq_len, 1)
        # 分母部分
        denominator = torch.exp(
            -math.log(10000) * torch.arange(0, embedding_dim, 2).float() / embedding_dim
        )

        pe[:, 0::2] = torch.sin(numerator * denominator)  # 偶数维度
        pe[:, 1::2] = torch.cos(numerator * denominator)  # 奇数维度

        # 增加一个维度, 变成 (1, seq_len, embedding_dim), 第一个维度表示batch_size
        pe = pe.unsqueeze(0)
        # ! 对位置编码进行缩放, 防止位置编码信息覆盖语义信息
        pe = pe * scale_factor
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        postional_embedding = self.pe[:, :x.size(1), :] # type: ignore
        return x + postional_embedding


if __name__ == "__main__":
    seq_len = 10
    embedding_dim = 16
