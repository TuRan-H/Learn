"""
b站教学视频: 【手把手带你实战HuggingFace Transformers-入门篇】基础组件之trainer

对文本分类任务代码进行优化:
使用transformers提供的trainer类简化训练流程
	- 使用了trainer类后, 不需要手动构建dataloader
	- 使用了trainer类后, 不需要自己写train函数
"""
import os, sys
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import evaluate
import json
import transformers
from torch.optim import Adam
from transformers import (
	TrainingArguments,
	AutoTokenizer, 
	AutoModelForSequenceClassification,
	PreTrainedTokenizerFast,
	DataCollatorWithPadding,
	Trainer
)
from datasets import load_dataset
from functools import partial

print(os.environ['TOKENIZERS_PARALLELISM'])


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


def compute_metrics(
		model_prediction:transformers.trainer_utils.EvalPrediction, 
		acc_metrics:evaluate.module.EvaluationModule,
		f1_metrics:evaluate.module.EvaluationModule
):
	"""
	用来传递给trainer类的 `compute_metrics` 参数的评估函数
	用来计算模型在验证集上的得分

	Args:
	---
		model_prediction (EvalPrediction): 模型的输出
			model_prediction是一个EvalPrediction类型, 该对象有两个属性
			- predictions: 模型输出的logits
			- label_ids: 真实的标签

		acc_metrics (EvaluationModule): 评价指标 accuracy

		f1_metrics (EvaluationModule): 评价指标 f1-score

	Returns:
	---
		返回一个字典, 该字典含有模型计算出的评价指标
	"""
	prediction, label = model_prediction.predictions, model_prediction.label_ids
	prediction = np.argmax(prediction, axis=-1)
	acc = acc_metrics.compute(predictions=prediction, references=label)
	f1 = f1_metrics.compute(predictions=prediction, references=label)
	acc.update(f1)

	return acc


if __name__ == '__main__':
	# 设置结果保存地址
	output_dir = "./results/text_classification_v3"
	# 导入模型和分词器
	tokenizer, model = load_model("model/rbt3")
	# 导入优化器
	optimizer = Adam(model.parameters(), lr=2e-5)
	# 读取数据
	dataset = load_dataset(path='csv', data_files="./dataset/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv", split='train')
	# 划分数据集
	# train_test_split()函数返回的是一个DatasetDict, 这个Dict中分别有train和test字段
	dataset = dataset.train_test_split(test_size=0.1)
	# 过滤数据, lamda函数的作用是接受一个样本, 假设这个样本中存在review字段则返回True, 否则返回False
	dataset = dataset.filter(lambda x: x['review'] is not None)
	# 映射数据, 对数据进行分词
	process_function = partial(process_example, tokenizer=tokenizer)
	dataset = dataset.map(process_function, batched=True, batch_size=1024, remove_columns=dataset['train'].column_names)

	train_dataset, valid_dataset = dataset['train'], dataset['test']

	training_args = TrainingArguments(
		output_dir=output_dir,
		per_device_train_batch_size=64,
		per_device_eval_batch_size=128,
		logging_steps=10,
		evaluation_strategy=transformers.trainer_utils.IntervalStrategy.EPOCH,
		save_strategy=transformers.trainer_utils.IntervalStrategy.EPOCH,
		learning_rate=2e-5,
		weight_decay=1e-2,
		metric_for_best_model='f1',
		load_best_model_at_end=True,
	)

	acc_metrics = evaluate.load('./utils/evaluate/metric_accuracy.py')
	f1_metrics = evaluate.load('./utils/evaluate/metric_f1.py')
	compute_metrics = partial(compute_metrics, acc_metrics=acc_metrics, f1_metrics=f1_metrics)
	trainer = Trainer(
		model=model, 
		args=training_args,
		data_collator=DataCollatorWithPadding(tokenizer),
		train_dataset=train_dataset,
		eval_dataset=valid_dataset,
		compute_metrics=compute_metrics,
	)

	# 模型开始训练
	trainer.train()

	# 模型评估
	eval_result = trainer.evaluate()
	with open(os.path.join(output_dir, "evaluate_result.json"), 'w') as fp:
		json.dump(eval_result, fp, indent=4)