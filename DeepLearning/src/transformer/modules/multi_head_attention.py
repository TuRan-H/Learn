"""Multi-Head Attention Module"""

import math
import torch

from torch import nn
from torch.nn import functional as F


class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        embedding_dim: int,
        dropout_rate: float,
        max_length: int,
    ) -> None:
        """
        Args:
            head_dim: 每个head的维度
            embedding_dim: 输入的embedding维度
            dropout_rate: dropout的概率
            max_length: 序列的最大长度
        """
        super().__init__()
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        self.k_proj = nn.Linear(self.embedding_dim, self.head_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.head_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.head_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.register_buffer("attention_mask", torch.tril(torch.ones(max_length, max_length)))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch_size, seq_len, emb_dim)
        """
        # k, q, v -> (batch_size, seq_len, head_dim)
        _, seq_length, _ = X.size()
        assert seq_length <= self.max_length, "Sequence length exceeds maximum length."

        k: torch.Tensor = self.k_proj(X)
        q: torch.Tensor = self.q_proj(X)
        v: torch.Tensor = self.v_proj(X)

        assert isinstance(self.attention_mask, torch.Tensor)

        # attention_weight: (batch_size, seq_len, seq_len)
        attention_score = q @ k.transpose(-1, -2)
        attention_score = attention_score.masked_fill(
            self.attention_mask[:seq_length, :seq_length] == 0, float("-inf")
        ) / math.sqrt(self.head_dim)
        attention_weight = F.softmax(attention_score, dim=-1)
        # 对attention_weight进行dropout, 防止模型过分关注某些token之间的相互关系, 导致过拟合
        attention_weight = self.dropout(attention_weight)

        # output -> (batch_size, seq_len, head_dim)
        output = attention_weight @ v

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, nums_head: int, head_dim: int, embedding_dim: int, dropout_rate: float, max_length: int) -> None:
        super().__init__()
        self.nums_head = nums_head
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(
                    head_dim=self.head_dim,
                    embedding_dim=self.embedding_dim,
                    dropout_rate=self.dropout_rate,
                    max_length=self.max_length,
                )
                for _ in range(self.nums_head)
            ]
        )
        self.o_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch_size, seq_len, emb_dim)
        """
        # 获取每个head的输出 (batch_size, seq_len, head_dim)
        heads_output = [head(X) for head in self.heads]
        # 将每个head的输出拼接起来
        heads_output = torch.concatenate(heads_output, dim=-1)
        output = self.o_proj(heads_output)
        output = self.dropout(output)

        return output


if __name__ == "__main__":
    x = torch.randn(2, 12, 512)
    mha = MultiHeadAttention(nums_head=8, head_dim=64, embedding_dim=512, dropout_rate=0.1, max_length=1024)
    y = mha(x)
    print(y.shape)
