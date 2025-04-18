"""
动手实现一个transformer模型, 用于文本生成任务

参考资料: 
* https://www.bilibili.com/video/BV1qWwke5E3K/?spm_id_from=333.1387.homepage.video_card.click&vd_source=41721633578b9591ada330add5535721

TODO:
* 实现RoPE(Positional Encoding变种)
* 实现SwiGLU (FFN中的激活函数变种)
* 实现RMSNorm (LayerNorm的变种)
* 实现FlashAttention (Attention的变种)
* 实现NSA (DeepSeek最新推出的Native Sparse Attention)
* 实现MLA (Multi Latent Attention)
* 实现KVCache
* 实现BEAM search
"""
import json
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
import tiktoken
from tqdm import tqdm


@dataclass
class TransformerConfig:
	"""
	创建配置参数类
	"""
	block_size: int = field(default=512, metadata={"help": "seq length"})
	batch_size: int = field(default=12, metadata={"help": "batch size"})
	vocab_size: int = field(default=50257, metadata={"help": "vocab size"})
	dropout_rate: float = field(default=0.1, metadata={"help": "dropout rate"})
	n_layer: int = field(default=6, metadata={"help": "the number of decoder layers"})
	n_head: int = field(default=12, metadata={"help": "the number of heads in MLA"})
	embedding_dim: int = field(default=768, metadata={"help": "the dimension of embeddings"})
	head_dim: int = field(default=64, metadata={"help": "the dimension of a head in MLA"})
	learning_rate: float = field(default=3e-4, metadata={"help": "learning rate"})
	T_max: int = field(default=1000, metadata={"help": "cosine annealing learning rate scheduler的超参"})
	device: str = field(default="cuda:0", metadata={"help": "device"})
	total_epoch: int = field(default=2, metadata={"help": "total epoch"})
	save_dir: str = field(default="./result/transformer", metadata={"help": "save model dir"})
	data_dir: str = field(default="./data/mobvoi_seq_monkey_general_open_corpus.jsonl", metadata={"help": "data dir"})

	def __post_init__(self):
		assert self.embedding_dim % self.n_head == 0, "n_embd must be divided by n_head"
		self.head_dim = self.embedding_dim // self.n_head


class SingleHeadAttention(nn.Module):
	def __init__(self, config: TransformerConfig) -> None:
		super().__init__()
		self.head_dim = config.head_dim
		self.embedding_dim = config.embedding_dim
		self.dropout_rate = config.dropout_rate
		self.k_proj = nn.Linear(self.embedding_dim, self.head_dim)
		self.q_proj = nn.Linear(self.embedding_dim, self.head_dim)
		self.v_proj = nn.Linear(self.embedding_dim, self.head_dim)
		self.dropout = nn.Dropout(self.dropout_rate)

		"""
		nn.Module.register_buffer(): 用来注册一些不需要计算梯度的张量, 这些张量可像参数一样被访问
		使用self.register_buffer("buffer名称", buffer值)即可注册一个buffer, 后买直接使用 `self.buffer名称` 即可访问

		seq_length的最大长度为block_size, 因此下三角矩阵的shape为 (block_size, block_size)
		值为1代表参与计算, 0代表不参与计算
		"""
		self.register_buffer(
			"attention_mask",
			torch.tril(torch.ones(config.block_size, config.block_size))
		)
	

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		# k, q, v -> (batch_size, seq_len, head_dim)
		batch_size, seq_length, embedding_dim = X.size()
		k: torch.Tensor = self.k_proj(X)
		q: torch.Tensor = self.q_proj(X)
		v: torch.Tensor = self.v_proj(X)

		# attention_weight -> (batch_size, seq_len, seq_len)
		attention_weight = q @ k.transpose(-1, -2)
		attention_weight = attention_weight.masked_fill(
			self.attention_mask[:seq_length, :seq_length] == 0,	# type: ignore
			float("-inf")
		) / math.sqrt(self.head_dim)
		attention_weight = F.softmax(attention_weight, dim=-1)
		# 对attention_weight进行dropout, 防止模型过分关注某些token之间的相互关系, 导致过拟合
		attention_weight = self.dropout(attention_weight)

		# output -> (batch_size, seq_len, head_dim)
		output = attention_weight @ v

		return output


class MultiHeadAttention(nn.Module):
	def __init__(self, config: TransformerConfig) -> None:
		super().__init__()
		self.n_head = config.n_head
		self.embedding_dim = config.embedding_dim
		self.dropout_rate = config.dropout_rate
		self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(self.n_head)])
		self.o_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
		self.dropout = nn.Dropout(self.dropout_rate)


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		# 获取每个head的输出 (batch_size, seq_len, head_dim)
		heads_output = [head(X) for head in self.heads]
		# 将每个head的输出拼接起来
		heads_output = torch.concatenate(heads_output, dim=-1)
		output = self.o_proj(heads_output)
		output = self.dropout(output)

		return output



class FeedForward(nn.Module):
	def __init__(self, config: TransformerConfig) -> None:
		super().__init__()
		self.embedding_dim = config.embedding_dim
		self.dropout_rate = config.dropout_rate
		self.net = nn.Sequential(
			nn.Linear(self.embedding_dim, 4 * self.embedding_dim),
			nn.GELU(),
			nn.Linear(4 * self.embedding_dim, self.embedding_dim),
			# ! FeedForward在过完两个线性层之后, 需要使用dropout, 防止过拟合
			nn.Dropout(self.dropout_rate)
		)


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		return self.net(X)


class DecoderBlock(nn.Module):
	def __init__(self, config: TransformerConfig) -> None:
		super().__init__()
		self.embedding_dim = config.embedding_dim
		self.mla = MultiHeadAttention(config)
		self.ffn = FeedForward(config)
		self.ln1 = nn.LayerNorm(self.embedding_dim)
		self.ln2 = nn.LayerNorm(self.embedding_dim)


	def forward(self, X: torch.Tensor) -> torch.Tensor:
		X = X + self.mla(self.ln1(X))
		X = X + self.ffn(self.ln2(X))

		return X


class Transformer(nn.Module):
	def __init__(self, config: TransformerConfig) -> None:
		super().__init__()
		self.embedding_dim = config.embedding_dim
		self.vocab_size = config.vocab_size
		self.block_size = config.block_size
		self.n_layer = config.n_layer
		self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
		self.positional_embedding = nn.Embedding(self.block_size, self.embedding_dim)
		self.decoder_blocks = nn.Sequential(
			*[DecoderBlock(config) for _ in range(self.n_layer)]
		)
		self.ln = nn.LayerNorm(self.embedding_dim)
		self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
		# tie weight, 即token_embedding和lm_head的权重共享
		# 这里lm_head是embedding_dim -> vocab_size的, 因此其weight的shape为(vocab_size, embedding_dim)
		# 和nn.Embeddging的weight的shape一致
		self.lm_head.weight = self.token_embedding.weight
		self.loss_function = nn.CrossEntropyLoss()

		self.apply(self._init_weights)


	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		if isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)


	def forward(self, X: torch.Tensor, targets: torch.Tensor | None = None):
		"""
		Args:
			X: (batch_size, seq_len)
			targets: (batch_size, seq_len)
		"""
		batch_size, seq_length = X.size()
		# X过token embedding, token_embedding -> (batch_size, seq_len, embedding_dim)
		token_embedding = self.token_embedding(X)
		# X过positional embedding, positional_embedding -> (batch_size, seq_len, embedding_dim)
		positional_embedding = self.positional_embedding(
			torch.arange(seq_length, device=X.device)
		)
		# token_embedding和positional_embedding相加
		embedding = token_embedding + positional_embedding
		output = self.decoder_blocks(embedding)
		output = self.ln(output)
		# logits -> (batch_size, seq_len, vocab_size)
		logits: Tensor = self.lm_head(output)

		if targets is None:
			loss = None
		else:
			batch_size, seq_length, vocab_size = logits.size()
			logits = logits.view(batch_size * seq_length, vocab_size)
			targets = targets.view(batch_size * seq_length)
			loss = self.loss_function(logits, targets)
		
		return logits, loss


	def generate(self, inputs: torch.Tensor, max_new_tokens: int):
		"""
		Args:
			# idx: (batch_size, seq_length) 经过tokenizer编码后的输入序列
			# max_new_tokens: 生成的最大token数量
		# """
		for _ in range(max_new_tokens):
			# 如果序列的长度大于block_size, 则只取最后block_size个token
			inputs = inputs if inputs.size(1) <= self.block_size else inputs[:, -self.block_size:]
			
			# 获取模型输出的logits, logits -> (batch_size, vocab_size)
			with torch.no_grad():
				logits, _ = self(inputs)
				logits = logits[:, -1, :]
			# 对模型输出的logits进行softmax
			probs = F.softmax(logits, dim=-1)
			# 从概率分布中采样下一个token, 这里使用的是多项式采样 (根据指定的概率分布进行采样)
			next_token = torch.multinomial(probs, num_samples=1)
			# 附加到原始输入序列的末尾, 作为下一次的输入
			inputs = torch.cat((inputs, next_token), dim=1)
		return inputs




class MyDataset(Dataset):
	def __init__(self, path, config: TransformerConfig) -> None:
		self.block_size = config.block_size
		self.tokenizer = tiktoken.get_encoding('gpt2')
		self.eos_token = "<|endoftext|>"
		self.eos_token_id = self.tokenizer.encode(self.eos_token, allowed_special="all")[0]
		self.max_line = 1000

		# 读取所有原始数据
		raw_data = []
		with open(path, 'r') as f:
			for i, line in enumerate(f):
				if i >= self.max_line: break
				raw_data.append(json.loads(line)['text'])

		# 将所有原始数据编码, 并且拼接为一个长序列
		full_encoded = []
		for text in raw_data:
			encoded = self.tokenizer.encode(text)
			full_encoded.extend(encoded + [self.eos_token_id])

		# 将长序列切分为多个block, 作为训练数据
		self.encoded_data = []
		bar = tqdm(range(0, len(full_encoded), self.block_size), desc="Encoding data")
		for i in range(0, len(full_encoded), self.block_size):
			# 每个block的长度为block_size + 1, [0: block_size]为inputs, [1: block_size + 1]为targets
			block = full_encoded[i: i + self.block_size + 1]
			# 如果最后一个block的大小不够, 使用eos_token_id作为padding
			if len(block) < self.block_size + 1:
				block += [self.eos_token_id] * (self.block_size + 1 - len(block))
			self.encoded_data.append(block)
			bar.update(1)


	def __len__(self):
		return len(self.encoded_data)


	def __getitem__(self, idx):
		block = self.encoded_data[idx]
		inputs = torch.tensor(block[:-1], dtype=torch.long)
		targets = torch.tensor(block[1:], dtype=torch.long)

		return inputs, targets


def train_epoch(
	model: Transformer,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	train_loader: DataLoader,
	val_loader: DataLoader,
	device: str,
	epoch: int
):
	# 首先将model设置为训练模式, 开启dropout
	model.train()
	total_loss = 0.0

	bar = tqdm(total=len(train_loader))
	for x, y in train_loader:
		# ! 将模型的梯度置零, 防止梯度累加. (tips: 只需要在loss.backward()之前置零即可)
		optimizer.zero_grad()
		x, y = x.to(device), y.to(device)
		logits, loss = model(x, y)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()

		bar.update(1)
		bar.set_description(f"Epoch: {epoch}, Train loss: {loss.item():.4f}")
	scheduler.step()

	return total_loss


def eval_epoch(
	model: Transformer,
	val_dataloader: DataLoader,
	device: str,
	epoch: int
):
	model.eval()
	total_loss = 0.0

	bar = tqdm(total=len(val_dataloader))
	with torch.no_grad():
		for x, y in val_dataloader:
			x, y = x.to(device), y.to(device)
			logits, loss = model(x, y)
			total_loss += loss.item()
			bar.update(1)
			bar.set_description(f"Epoch: {epoch}, Eval loss: {loss.item():.4f}")

	return total_loss


def train(
	config: TransformerConfig,
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
			model,
			optimizer,
			scheduler,
			train_dataloader,
			val_dataloader,
			config.device,
			epoch
		)
		val_loss = eval_epoch(
			model,
			val_dataloader,
			config.device,
			epoch
		)
		print(f"Epoch: {epoch}, Train loss: {train_loss / len(train_dataloader)}, Eval loss: {val_loss / len(val_dataloader)}")

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



if __name__ == '__main__':
	torch.manual_seed(1024)
	config = TransformerConfig()
	dataset = MyDataset(config.data_dir, config)
	train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
	train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
	model = Transformer(config)

	# train(config, model, train_dataloader, val_dataloader)

	checkpoint = torch.load("./result/transformer/transformer.pth")
	model.load_state_dict(checkpoint['model_state_dict'])
	tokenizer = tiktoken.get_encoding('gpt2')
	input_ids = tokenizer.encode("who are you")
	input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
	outputs = model.generate(input_ids, 100)
	outputs = outputs.tolist()
	outputs = tokenizer.decode_batch(outputs)