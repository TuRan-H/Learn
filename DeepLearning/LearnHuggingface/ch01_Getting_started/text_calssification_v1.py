"""
b站教学视频: 【手把手带你实战HuggingFace Transformers-入门篇】基础组件之Datasets
视频地址: https://www.bilibili.com/video/BV1Ph4y1b76w/?spm_id_from=333.788.top_right_bar_window_history.content.click&vd_source=41721633578b9591ada330add5535721

对文本分类任务代码进行优化
使用load_dataset来优化数据集的读取

"""
# ************************************************** 修改cwd
from curses import flash
import os, sys
os.chdir("/private/TuRan/LEARN/DeepLearning")
sys.path.append("/private/TuRan/LEARN/DeepLearning")
# ************************************************** 修改cwd

from sympy import false
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from transformers import (
	AutoTokenizer, 
	AutoModelForSequenceClassification,
	PreTrainedTokenizerFast,
	PreTrainedModel,
	DataCollatorWithPadding
)
from datasets import load_dataset
from tqdm import tqdm
from functools import partial


def process_example(examples, tokenizer:PreTrainedTokenizerFast):
	"""
	用于dataset.map(), 对数据集中的所有元素进行处理
	"""
	tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
	tokenized_examples['label'] = examples['label']

	return tokenized_examples


def load_model(model_name_or_path:str):
	"""
	导入模型和tokenizer
	"""
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
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	epoch = kwargs.pop('epoch')
	global_step = 0

	# 将模型调整至训练模式
	model.train()
	for ep in range(epoch):
		bar = tqdm(total=len(dataloader), desc=f"epoch = {ep+1}")
		for input in dataloader:
			# 将input放到GPU上
			input = {k:v.to(device) for k, v in input.items()}
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
				print(f"loss is {output['loss']} ,global step is {global_step}")

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
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	acc_num = 0
	model.to(device)
	model.eval()
	print("start evaluate")
	with torch.inference_mode():
		for input in tqdm(dataloader):
			input = {k:v.to(device) for k, v in input.items()}
			output = model(**input)
			# output['logits']是一个 (batch_size , num_class) 的tensor
			predictions = torch.argmax(output['logits'], dim=-1)
			
			# 计算准确率
			acc_num += (predictions.long() == input['labels'].long()).float().sum()
	return acc_num / len(dataloader.dataset)


if __name__ == '__main__':
	# 导入模型和分词器
	tokenizer, model = load_model("model/rbt3")

	# 导入优化器
	optimizer = Adam(model.parameters(), lr=2e-5)

	# 读取数据
	dataset = load_dataset(path='csv', data_files="dataset/ChnSentiCorp_htl_all.csv", split='train')
	# 划分数据集
	# train_test_split()函数返回的是一个DatasetDict, 这个Dict中分别有train和test字段
	dataset = dataset.train_test_split(test_size=0.1)
	# 过滤数据, lamda函数的作用是接受一个样本, 假设这个样本中存在review字段则返回True, 否则返回False
	dataset = dataset.filter(lambda x: x['review'] is not None)
	# 映射数据
	process_function = partial(process_example, tokenizer=tokenizer)
	dataset = dataset.map(process_function, batched=True, batch_size=1024, remove_columns=dataset['train'].column_names)

	train_dataset, valid_dataset = dataset['train'], dataset['test']

	train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
	valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))

	# 开始训练
	train(tokenizer, model, optimizer, train_dataloader, epoch=3)

	# 开始评估, 计算评价指标
	metrics = evaluate(tokenizer, model, valid_dataloader)

	# play-ground
	idtolabel = {
		0: "不好",
		1: "好"
	}

	model.to('cpu')
	input = "优的酒店"
	input = tokenizer(input, return_tensors='pt')
	output = model(**input)
	prediction = idtolabel[torch.argmax(output['logits'], dim=-1).item()]
	print(prediction)