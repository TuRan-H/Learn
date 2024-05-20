"""
b站教学视频: 【手把手带你实战HuggingFace Transformers-入门篇】基础组件之Evaluate
视频地址: https://www.bilibili.com/video/BV1uk4y1W7tK/?spm_id_from=333.788&vd_source=41721633578b9591ada330add5535721

对文本分类任务代码进行优化: 使用Huggingface的evaluate来进行评估, 简化代码逻辑

"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from transformers import (
	AutoTokenizer, 
	AutoModelForSequenceClassification,
	PreTrainedTokenizerFast,
	PreTrainedModel,
	DataCollatorWithPadding
)
import evaluate
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


def train(model:PreTrainedModel, optimizer:torch.optim.Adam, train_dataloader:DataLoader, valid_dataloader:DataLoader ,total_epoch:int):
	"""
	Trains a text classification model.

	Args:
	---
		model (PreTrainedModel): The text classification model.
		optimizer (torch.optim.Adam): The optimizer used to update the model's parameters.
		train_dataloader (DataLoader): The data loader for training data, used to process a batch of training data.
		valid_dataloader (DataLoader): The data loader for validation data, used to process a batch of validation data.
		total_epoch (int): The total number of training epochs.

	Returns:
	---
		None
	"""
	# Move the model to GPU if available
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)

	# Set the model to training mode
	model.train()
	for epoch in range(total_epoch):
		bar = tqdm(total=len(train_dataloader), desc=f"epoch = {epoch+1}")
		for input in train_dataloader:
			# Move input to GPU
			input = {k:v.to(device) for k, v in input.items()}
			# 将输入传递给模型，计算预测结果
			# 如果输入包含`labels`字段，模型将自动计算损失
			output = model(**input)
			# Backpropagation
			output.loss.backward()
			# Gradient update
			optimizer.step()
			# 在每个batch训练结束后, 置零梯度, 防止梯度累计
			optimizer.zero_grad()

			bar.update(1)
		# 每个epoch结束后计算metrics, 并输出
		print(compute_metrics(model, valid_dataloader))



def compute_metrics(model:PreTrainedModel, dataloader:DataLoader):
	"""
	评估函数, 用来计算模型在验证集上的得分

	Args:
	---
		model: 训练的模型
		dataloader: 用来进行验证的验证集dataloader
	
	Returns:
	---
		输出模型在验证集上的metrics
	"""
	device = model.device
	# 导入评价指标
	metrics = evaluate.combine(['accuracy', 'f1'])

	# 给定验证集中所有数据, 计算metrics
	for batch in dataloader:
		input = {k:v.to(device) for k, v in batch.items()}
		output = model(**input)
		prediction = torch.argmax(output.logits, dim=1).reshape([-1])

		metrics.add_batch(references=input['labels'].long(), predictions=prediction.long())

	metrics_result = metrics.compute()

	return metrics_result


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
	# 映射数据, 对数据进行分词
	process_function = partial(process_example, tokenizer=tokenizer)
	dataset = dataset.map(process_function, batched=True, batch_size=1024, remove_columns=dataset['train'].column_names)

	train_dataset, valid_dataset = dataset['train'], dataset['test']

	# 创建dataloader
	train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
	valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))

	# 开始训练
	train(model, optimizer, train_dataloader, valid_dataloader, 3)

	# 开始评估, 计算评价指标
	metrics = compute_metrics(model, valid_dataloader)


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