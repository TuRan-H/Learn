"""
self-attention

参考资料: https://www.bilibili.com/video/BV19YbFeHETz/?t=1083.2&vd_source=41721633578b9591ada330add5535721
"""

import math
import torch
from torch import nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 728, dropout_rate: float = 0.1) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attention_dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch_size, seq_length, hidden_dim)
            attention_mask: 注意力掩码

        Returns:
            output: (batch_size, seq_length, hidden_dim)
        """
        # 通过一个大矩阵完成QKV的映射, QKV -> (batch_size, seq_length, hidden_dim * 3)
        QKV = self.proj(X)

        # 分割QKV
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)

        # 计算attention得分
        # attention_weight -> (batch_size, seq_length, seq_length)
        attention_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)

        if attention_mask is not None:
            # 如果attention_weight的shape为 (batch_size, seq_length)
            # 则将attention_weight扩展为 (batch_size, seq_length, seq_length)
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask.unsqueeze(1)
                attention_mask = attention_mask.expand(-1, attention_weight.shape[-1], -1)
            # 对attention_weight进行填充
            attention_weight = attention_weight.masked_fill(attention_mask, float('-inf'))

        attention_weight = torch.softmax(attention_weight, -1)

        # 对attention得分进行dropout
        attention_weight = self.attention_dropout(attention_weight)

        # output -> (batch_size, seq_length, hidden_dim)
        output = attention_weight @ V

        return output


if __name__ == '__main__':
    x = torch.randn((16, 64, 128))
    # 创建伯努利随机掩码
    mask = torch.bernoulli(torch.ones(x.shape[:-1]), 0.8).bool()

    model = SelfAttention(128)
    y = model(x, mask)
    print(y.shape)
