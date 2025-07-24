"""
动手实现: 【手把手带你实战HuggingFace Transformers-实战篇】实战演练之生成式对话机器人
对bloom模型进行指令微调, 使其能够做到单轮对话


视频地址: https://www.bilibili.com/video/BV11r4y197Ht/?spm_id_from=333.788&vd_source=41721633578b9591ada330add5535721
使用的backbone model: Langboat/bloom-389m-zh
使用的数据集: alpaca_data_zh
"""
import os, sys
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
from transformers.trainer import Trainer, TrainingArguments


def process_function(example, tokenizer:PreTrainedTokenizerFast):
	MAX_LENGTH = 256
	input_ids, attention_mask, labels = [], [], []
	# 对输入样本进行分词
	instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
	response = tokenizer(example['output']+tokenizer.eos_token)

	# 计算input_ids
	input_ids = instruction['input_ids'] + response['input_ids']
	# 计算attention_mask
	attention_mask = instruction['attention_mask'] + response['attention_mask']
	# 计算labels
	labels = [-100] * len(instruction['input_ids']) + response['input_ids']

	if len(input_ids) > MAX_LENGTH:
		input_ids = input_ids[:MAX_LENGTH]
		attention_mask = attention_mask[:MAX_LENGTH]
		labels = labels[:MAX_LENGTH]
	
	return {
		'input_ids': input_ids,
		'attention_mask': attention_mask,
		'labels': labels
	}



if __name__ == "__main__":
	# 导入数据集
	dataset = Dataset.load_from_disk("./dataset/alpaca_data_zh")

	# 导入模型和分词器
	model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-389m-zh")
	tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-389m-zh")
	
	# remove_cloums: 进行map后自动删除某些字段
	dataset = dataset.map(partial(process_function, tokenizer=tokenizer), remove_columns=dataset.column_names)

	# 定义TrainingArgument和Trainer
	training_args = TrainingArguments(
		output_dir="./results/chatbot",
		per_device_train_batch_size=16,
		gradient_accumulation_steps=2,
		logging_steps=100,
		num_train_epochs=2,
		save_strategy="epoch"
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset,
		data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
	)
	
	trainer.train()

	# 进行推理
	from transformers import pipeline
	inputs = "Human: {}\n{}".format("怎样才能够在考试时获取好成绩", "").strip() + "\n\nAssistant: "
	pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
	result = pipe(inputs, max_length=256, do_sample=True)
	print(result[0]['generated_text'])
	