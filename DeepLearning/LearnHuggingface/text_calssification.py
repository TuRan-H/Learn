"""
b站教学视频: 【手把手带你实战HuggingFace Transformers-入门篇】基础组件之Model（下）BERT文本分类代码实例
视频地址: https://www.bilibili.com/video/BV18T411t7h6/?spm_id_from=333.788&vd_source=41721633578b9591ada330add5535721

使用pytorch框架实现文本分类任务
对预训练的语言模型进行微调

使用的backbone模型: hfl/rbt3 
	https://huggingface.co/hfl/rbt3
用来fine-tune的数据集: 
	ChnSentiCorp_htl_all.csv
	下载地址: https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv

总体流程
	1. 导入数据集
	2. 对数据集进行预处理
	3. 导入模型
	4. 对模型进行fine-tune
	5. 使用微调后的模型, 进行inference, 并compute_metrics
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
	PreTrainedTokenizerFast,
	PreTrainedModel
)
from torch.optim import Adam
from tqdm import tqdm

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
			
			acc_num += (predictions.long() == input['labels'].long()).float().sum()
	return acc_num / len(dataloader.dataset)





if __name__ == '__main__':
	# 导入模型和分词器
	tokenizer, model = load_model("model/rbt3")

	# 导入优化器
	optimizer = Adam(model.parameters(), lr=2e-5)

	# 划分数据集
	all_dataset = MyDataset("dataset/ChnSentiCorp_htl_all.csv")
	train_dataset, valid_dataset = random_split(all_dataset, [0.8, 0.2])

	# 实例化DataCollator
	datacollator = MyDataCollator(tokenizer)

	train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=datacollator, shuffle=True)
	valid_dataloader = DataLoader(valid_dataset, batch_size=64, collate_fn=datacollator)


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
	input = "很好的酒店"
	input = tokenizer(input, return_tensors='pt')
	output = model(**input)
	prediction = idtolabel[torch.argmax(output['logits'], dim=-1).item()]
	print(prediction)