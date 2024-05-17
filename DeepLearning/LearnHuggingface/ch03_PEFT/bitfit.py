"""
代码: 【手把手带你实战HuggingFace Transformers-高效微调篇】参数高效微调与BitFit实战

主要内容:
	冻结网络中所有的非 `bias` 的参数, 进行训练
	采用这种方式能够极大地节约显存


视频地址: https://www.bilibili.com/video/BV1Xu4y1k7Ls/?spm_id_from=333.788&vd_source=41721633578b9591ada330add5535721
backbone: Langboat/bloom-1b4-zh
corpus: alpaca_data_zh
"""
# %%
from functools import partial
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from LearnHuggingface.ch02_NLP_Tasks.generative_chatbot import process_function


# %%
if __name__ == "__main__":
	# 导入模型和分词器
	model = AutoModelForCausalLM.from_pretrained('./model/bloom-1b4-zh')
	tokenizer = AutoTokenizer.from_pretrained('./model/bloom-1b4-zh')

	# 导入数据集, 并对数据集进行预处理
	dataset = Dataset.load_from_disk('./dataset/alpaca_data_zh')
	dataset = dataset.map(partial(process_function, tokenizer=tokenizer), remove_columns=dataset.column_names)

	# %%
	# bitfit 冻结模型中的参数
	num_param = 0
	for name, param in model.named_parameters():
		if 'bias' not in name:
			param.requires_grad  = False	# ! 属性名不要拼写错误
		else:
			num_param += param.numel()

	# %%
	# 实例化trainer
	training_args = TrainingArguments(
		output_dir="./results/chatbot",
		per_device_train_batch_size=8,
		gradient_accumulation_steps=4,
		logging_steps=10,
		save_strategy='epoch',
		num_train_epochs=2
	)

	trainer = Trainer(
		model=model,
		tokenizer=tokenizer,
		train_dataset=dataset,
		data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
		args=training_args
	)

	# %%
	# 开始训练
	trainer.train()

	# %%
	# inference
	inputs = "Human: {}\n{}".format("怎样才能够在考试时获取好成绩", "").strip() + "\n\nAssistant: "
	pipe = pipeline('text-gereration', model=model, tokenizer=tokenizer, device='0')
	result = pipe(inputs, max_length=256, do_sample=True)
	print(result[0]['generated_text'])