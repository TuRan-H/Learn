"""
【手把手带你实战HuggingFace Transformers-实战篇】实战演练之命名实体识别

命名实体识别任务
使用peoples_daily_ner数据集fine-tune chinese-macbert-base模型

课程链接 https://www.bilibili.com/video/BV1gW4y197CT/?spm_id_from=333.788&vd_source=41721633578b9591ada330add5535721

使用的数据集: https://huggingface.co/datasets/peoples_daily_ner
使用的backbone: https://huggingface.co/hfl/chinese-macbert-base
使用的metrics: https://huggingface.co/spaces/evaluate-metric/seqeval
"""
import evaluate
from functools import partial
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizer
from transformers.trainer import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForTokenClassification
import numpy as np



def process_function(examples, tokenizer:PreTrainedTokenizer):
	"""
	对数据集进行预处理

	Args:
	---
		examples: 原始数据集的一个batch
			examples中的每一个元素分别有两个字段
			- tokens: 列表, 每个元素都是一个中文字符, 所有字符构成了一句话
			- ner_tags: 列表, 每个元素表示一个中文字符所对应的标签, 采用整数形式的BIO标签
		tokenizer: 分词器
	
	Returns:
	---
		返回toeknized_example
		tokenized_example中含有三个主要的key
		- input_ids
		- attention_mask
		- labels
	"""
	# TODO `is_split_into_words`: 

	tokenized_examples = tokenizer(examples['tokens'], max_length=128, truncation=True, is_split_into_words=True)

	# 一个batch中所有元素的label所组成的列表
	labels = []
	for i, label in enumerate(examples['ner_tags']):
		temp = []
		# * 将每一个token的label通过word_ids对其到真正的label
		# word_ids是tokenized后的token和原始输入中的words之间的对照表
		word_ids = tokenized_examples.word_ids(batch_index=i)
		for ids in word_ids:
			if ids == None:
				# 如果ids为None就表示当前这个token可能是 [CLS] 或者 [SEP] token, 将其label设置为 -100, 不参与loss计算
				temp.append(-100)
			else:
				temp.append(label[ids])
		labels.append(temp)
	tokenized_examples['labels'] = labels

	return tokenized_examples


def compute_metrics(model_predictions, id2label:list, metrics:str):
	"""
	评估函数, 用来计算评价指标

	Args:
	---
		model_predictions: 模型的预测值, 以batch的形式传入
		id2label: 标签列表
		metrics: 评价指标的名称
	
	Returns:
	---
		评价指标
	"""
	metrics = evaluate.load(metrics)
	predictions, labels = model_predictions.predictions, model_predictions.label_ids
	predictions = np.argmax(predictions, axis=-1)

	true_predictions = []
	true_labels = []
	for prediction, label in zip(predictions, labels):
		true_predictions.append([id2label[p] for p, l in zip(prediction, label) if l != -100])
		true_labels.append([id2label[l] for p, l in zip(prediction, label) if l != -100])

	metrics.add_batch(references=true_labels, predictions=true_predictions)
	result = metrics.compute(mode="strict", scheme="IOB2")

	return result






if __name__ == "__main__":
	# * 导入数据集
	# dataset = DatasetDict.load_from_disk("./dataset/peoples_daily_ner")
	dataset = load_dataset('peoples_daily_ner')
	id2label = dataset['train'].features['ner_tags'].feature.names


	# * 导入model和tokenizer
	# model = AutoModelForTokenClassification.from_pretrained('./model/chinese-macbert-base', num_labels=)
	model = AutoModelForTokenClassification.from_pretrained('./model/chinese-macbert-base', num_labels=len(id2label))
	tokenizer = AutoTokenizer.from_pretrained('./model/chinese-macbert-base')

	# * 对数据集进行预处理
	# ! dataset.map()不是原地方法, 必须要使用一个变量接受它
	dataset = dataset.map(partial(process_function, tokenizer=tokenizer), batch_size=64, batched=True)


	train_dataset, eval_dataset = dataset['train'], dataset['test']

	args = TrainingArguments(
		per_device_train_batch_size=32,
		per_device_eval_batch_size=64,
		logging_strategy='steps',
		logging_steps=100,
		evaluation_strategy="epoch",
		save_strategy="epoch",
		num_train_epochs=2,
		output_dir="./results/token_classification",
		load_best_model_at_end=True,
	)

	trainer = Trainer(
		model = model,
		tokenizer=tokenizer,
		args=args,
		data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),		# ! 这里不能传递一个类, 这里需要传递一个类的实例
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=partial(compute_metrics, id2label=id2label, metrics="seqeval")
	)

	trainer.train()

	trainer.evaluate()