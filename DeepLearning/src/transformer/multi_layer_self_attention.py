"""
实现transformer中的多头注意力机制

参考资料: https://www.bilibili.com/video/BV19mxdeBEbu?spm_id_from=333.788.videopod.sections&vd_source=41721633578b9591ada330add5535721
"""
import math
import torch
from torch import nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
	def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float = 0.1) -> None:
		super().__init__()
		self.hidden_dim = hidden_dim
		self.head_num = head_num
		self.head_dim = hidden_dim // head_num

		self.q_proj = nn.Linear(self.hidden_dim, self.head_num * self.head_dim)
		self.k_proj = nn.Linear(self.hidden_dim, self.head_num * self.head_dim)
		self.v_proj = nn.Linear(self.hidden_dim, self.head_num * self.head_dim)
		self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

		self.attention_dropout = nn.Dropout(dropout_rate)
		
	
	def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor: # type: ignore
		"""
		Args:
			X - (batch_size, seq_length, hidden_dim)
		
		Returns:
			output - (batch_size, seq_length, hidden_dim)
		"""
		batch_size, seq_length, _ = X.size()

		# Q, K, V - (batch_size, seq_length, head_num * head_dim)
		Q, K, V = self.q_proj(X), self.k_proj(X), self.v_proj(X)
		# 这里要将每个Q, K, V的输出进行切割, 提取出每个注意力头的内容
		# q_state - (batch_size, head_num, seq_length, head_dim)
		q_state: torch.Tensor = Q.view(batch_size, seq_length, self.head_num, self.head_dim).permute(0, 2, 1, 3)
		k_state: torch.Tensor = K.view(batch_size, seq_length, self.head_num, self.head_dim).permute(0, 2, 1, 3)
		v_state: torch.Tensor = V.view(batch_size, seq_length, self.head_num, self.head_dim).permute(0, 2, 1, 3)

		# attention_weight - (batch_size, head_num, seq_length, seq_length)
		attention_weight = q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim)

		# 使用attention_mask填充attention_weight
		if attention_mask is not None:
			# attention_mask - (batch_size, seq_length)
			# 这里对attention_mask进行处理, 使其匹配attention_weight的大小
			if len(attention_mask.size()) == 2:
				attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, seq_length)
				attention_mask = attention_mask.unsqueeze(1).expand(-1, self.head_num, -1, -1)
			attention_weight = attention_weight.masked_fill(attention_mask, float('-inf'))
		
		# attention_weight过softmax和dropout
		attention_weight = torch.softmax(attention_weight, -1)
		attention_weight: torch.Tensor = self.attention_dropout(attention_weight)

		# output - (batch_size, head_num, seq_length, head_dim)
		output = attention_weight @ v_state
		# 这里对output进行transpose之后会导致张量在内存中不再连续, 但是view操作仅支持连续的张量
		# 因此要么使用contiguous要么使用reshape代替view
		output = output.transpose(1, 3).contiguous().view(batch_size, seq_length, self.head_num * self.head_dim)
		# 多头注意力的输出必须要过一层投影层, 用来处理多个头concate起来的信息
		output = self.output_proj(output)

		return output
	
	def __call__(self, *args, **kwds) -> torch.Tensor:
		return super().__call__(*args, **kwds)


if __name__ == '__main__':
	attention_mask = (
		torch.tensor(
			[
				[0, 1],
				[0, 0],
				[1, 0],
			]
		)
	).bool()

	x = torch.rand(3, 2, 128)
	net = MultiHeadAttention(128, 8)
	y = net(x, attention_mask)
	print(y.shape)