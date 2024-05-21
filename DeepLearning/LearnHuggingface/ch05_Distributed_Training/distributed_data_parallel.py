"""
b站教学视频: 【手把手带你实战HuggingFace Transformers-分布式训练篇】分布式数据并行原理与应用

代码流程: 
	1. 使用pytorch框架对预训练的语言模型进行微调, 实现文本的情感分类任务
	2. 使用Distributed data parallel进行分布式训练

backbone: https://huggingface.co/hfl/rbt3
corpus: https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv

注意: 本篇代码需要使用 `torchrun` 来运行
"""
import os
import pandas as pd
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from tqdm import tqdm
from transformers import (
	AutoTokenizer,
	PreTrainedTokenizerFast,
	PreTrainedModel,
	AutoModelForSequenceClassification
)

class MyDataset(Dataset):
	def __init__(self, dataset_path: str = None) -> None:
		"""
		This class represents a custom dataset for text classification.
		Args:
		---
			dataset_path (str): The path to the dataset file.
		"""
		super().__init__()
		self.data = pd.read_csv(dataset_path)
		self.data = self.data.dropna()

	def __getitem__(self, index):
		return self.data.iloc[index]['review'], self.data.iloc[index]['label']

	def __len__(self):
		return len(self.data)


class MyDataCollator:
	def __init__(self, tokenizer:PreTrainedTokenizerFast) -> None:
		"""
		DataCollator，用于处理原始数据批次并生成模型输入。

		Args:
		---
			tokenizer (PreTrainedTokenizerFase): 预训练的分词器
		"""
		self.tokenizer = tokenizer
	
	def __call__(self, batch):
		"""
		获取模型一个batch的输入, 将这个输入进行处理

		Args:
		---
			batch: 一个batch的输入数据
		"""
		tokenizer = self.tokenizer
		instances, labels = zip(*batch)
		inputs = tokenizer(instances, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
		inputs['labels'] = torch.tensor(labels)

		return inputs


def load_model(model_name_or_path:str):
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

	return tokenizer, model


def train(tokenizer:PreTrainedTokenizerFast, model:PreTrainedModel, optimizer:torch.optim.Adam, dataloader:DataLoader, **kwargs):
	"""
	训练函数，用于训练文本分类模型。

	Args:
	---
		tokenizer (PreTrainedTokenizerFast): 用于对文本进行分词和编码的tokenizer。
		model (PreTrainedModel): 文本分类模型。
		optimizer (torch.optim.Adam): 优化器，用于更新模型的参数。
		dataloader (DataLoader): 数据加载器，用于对一个batch的数据进行处理。
		**kwargs: 其他可选参数。
			epoch: 用于确定epoch次数
	"""
	# 将model放到GPU上
	epoch = kwargs.pop('epoch')
	global_step = 0

	# 将模型调整至训练模式
	model.train()
	for ep in range(epoch):
		bar = tqdm(total=len(dataloader), desc=f"rank = {os.environ['LOCAL_RANK']}, epoch = {ep+1}")
		for input in dataloader:
			# 将input放到GPU上
			input = {k:v.to(int(os.environ['LOCAL_RANK'])) for k, v in input.items()}
			# 将input送给model, 计算prediciotns, 如果input中存在 `labels`字段,  model会自动计算其loss
			output = model(**input)
			# 反向传播
			output.loss.backward()
			# 梯度更新
			optimizer.step()
			# 在每一个batch的梯度更新结束后, 置零梯度, 防止梯度累计
			optimizer.zero_grad()
			global_step += 1

			if global_step % 100 == 0:
				print(f"loss is {output['loss']}, global step is {global_step}")

			bar.update(1)
		




def evaluate(tokenizer:PreTrainedTokenizerFast, model:PreTrainedModel, dataloader:DataLoader):
	"""
	Evaluate the performance of a text classification model on a given dataset.

	Args:
	---
		tokenizer (PreTrainedTokenizerFast): The tokenizer used to preprocess the input data.
		model (PreTrainedModel): The text classification model to evaluate.
		dataloader (DataLoader): The data loader that provides the evaluation dataset.

	Returns:
	---
		float: The accuracy of the model on the evaluation dataset.
	"""
	acc_num = 0
	model.eval()
	print("start evaluate")
	with torch.inference_mode():
		for input in tqdm(dataloader):
			input = {k:v.to(int(os.environ['LOCAL_RANK'])) for k, v in input.items()}
			output = model(**input)
			# output['logits']是一个 (batch_size , num_class) 的tensor
			predictions = torch.argmax(output['logits'], dim=-1)
			
			acc_num += int((predictions.long() == input['labels'].long()).float().sum())
	return int(acc_num) / len(dataloader.dataset)





if __name__ == '__main__':
	# * 设置distributed的后端
	distributed.init_process_group(backend="nccl")

	# 导入模型和分词器
	tokenizer, model = load_model("model/rbt3")

	# * 使用DDP包装模型
	# ! 注意, 现将模型导入到某张卡后, 再去包装模型
	model.to(int(os.environ['LOCAL_RANK']))
	model = DistributedDataParallel(model)

	# 导入优化器
	optimizer = Adam(model.parameters(), lr=2e-5)

	# 划分数据集
	all_dataset = MyDataset("dataset/ChnSentiCorp_htl_all.csv")
	train_dataset, valid_dataset = random_split(all_dataset, [0.8, 0.2])

	# 实例化DataCollator
	datacollator = MyDataCollator(tokenizer)

	# * 修改DataLoader的sampler为DistributedSampler, 使得不同的进程能够获取
	train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=datacollator, sampler=DistributedSampler(train_dataset))
	valid_dataloader = DataLoader(valid_dataset, batch_size=64, collate_fn=datacollator, sampler=DistributedSampler(valid_dataset))

	# 开始训练
	train(tokenizer, model, optimizer, train_dataloader, epoch=3)

	# 开始评估, 计算评价指标
	metrics = evaluate(tokenizer, model, valid_dataloader)
	print(metrics)

	print("hello world")