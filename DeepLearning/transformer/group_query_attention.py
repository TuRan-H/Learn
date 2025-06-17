"""
实现transformer中的GQA (Group Query Attention)

参考资料: http://arxiv.org/abs/2305.13245
"""

import math
import torch
from torch import nn as nn


class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_kv_head, attention_dropout_rate=0.1) -> None:
        super().__init__()

        assert hidden_dim % nums_head == 0, "hidden dim需要能够被query的头整除"
        assert nums_head % nums_kv_head == 0, "query的头能够被K, V的头整除"

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_kv_head = nums_kv_head
        self.head_dim = self.hidden_dim // self.nums_head

        self.q_proj = nn.Linear(self.hidden_dim, self.nums_head * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.nums_kv_head * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.nums_kv_head * self.head_dim)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.attention_droput = nn.Dropout(attention_dropout_rate)

    def __call__(self, *args: torch.Tensor, **kwds: torch.Tensor) -> torch.Tensor:
        return super().__call__(*args, **kwds)

    def forward(self, X: torch.Tensor, attention_mask=None):
        """
        Args:
                X: (batch_size, seq_length, hidden_dim)
        """
        batch_size, seq_length, _ = X.size()

        Q: torch.Tensor = self.q_proj(X)  # (batch_size, seq_length, nums_head * head_dim)
        K: torch.Tensor = self.k_proj(X)  # (batch_size, seq_length, nums_kv_head * head_dim)
        V: torch.Tensor = self.v_proj(X)

        Q = Q.view(batch_size, seq_length, self.nums_head, self.head_dim).transpose(
            1, 2
        )  # (batch_size, nums_head, seq_length, head_dim)
        K = K.view(batch_size, seq_length, self.nums_kv_head, self.head_dim).transpose(
            1, 2
        )  # (batch_size, nums_kv_head, seq_length, head_dim)
        V = V.view(batch_size, seq_length, self.nums_kv_head, self.head_dim).transpose(1, 2)

        # repeat_interleave: 元素级别的复制, 在dim=1上复制self.nums_head // self.nums_kv_head次
        K = K.repeat_interleave(self.nums_head // self.nums_kv_head, 1)
        V = V.repeat_interleave(self.nums_head // self.nums_kv_head, 1)

        attention_score = (
            Q @ V.transpose(-1, -2) / math.sqrt(self.head_dim)
        )  # (batch_size, nums_head, seq_length, seq_length)
        attention_score = torch.softmax(attention_score, -1)
        attention_score: torch.Tensor = self.attention_droput(attention_score)

        output = attention_score @ V  # (batch_size, nums_head, seq_length, head_dim)
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        )  # (batch_size, seq_length, nums_head * head_dim)
        output = self.o_proj(output)

        return output


if __name__ == '__main__':
    x = torch.rand(3, 2, 128)
    gqa = GroupQueryAttention(128, 8, 4)
    y = gqa(x)
    print(y.shape)
