"""
代码: 【手把手带你实战HuggingFace Transformers-高效微调篇】LoRA 原理与实战

使用lora包裹模型, 进行模型的训练, 在训练结束后, 合并lora模型和原模型

视频地址: https://www.bilibili.com/video/BV13w411y7fq/?spm_id_from=333.788&vd_source=41721633578b9591ada330add5535721

backbone: Langboat/bloom-1b4-zh
corpus: alpaca_data_zh
"""
import os
from functools import partial
from datasets import Dataset
from sympy import true
from torch import _test_autograd_multiple_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers.trainer import Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from LearnHuggingface.ch02_NLP_Tasks.generative_chatbot import process_function




if __name__ == "__main__":
	os.environ["TOKENIZERS_PARALLELISM"] = "false"

	# 导入模型和分词器
	model = AutoModelForCausalLM.from_pretrained('./model/bloom-1b4-zh')
	tokenizer = AutoTokenizer.from_pretrained('./model/bloom-1b4-zh')

	# 导入数据集, 并对数据集进行预处理
	dataset = Dataset.load_from_disk("./dataset/alpaca_data_zh")
	dataset = dataset.map(partial(process_function, tokenizer=tokenizer), remove_columns=dataset.column_names)

	# 创建lora config, 控制lora的行为
	config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		target_modules=".*query_key_value.*",
		modules_to_save=["word_embeddings"]
	)

	# 使用lora包装模型
	lora_model = get_peft_model(model, config)


	# 创建training_args和trainer
	training_args = TrainingArguments(
		output_dir="./results/chatbot_lora",
		per_device_train_batch_size=8,
		gradient_accumulation_steps=4,
		logging_steps=10,
		num_train_epochs=2
	)

	trainer = Trainer(
		model=lora_model,
		args=training_args,
		train_dataset=dataset,
		data_collator=DataCollatorForSeq2Seq(tokenizer)
	)

	# 开始训练
	trainer.train()

	# 合并模型
	model = lora_model.merge_and_unload(model)


	# 保存模型
	model.save_pretrained('./results/chatbot_lora/best_model')


	# 使用合并后的模型进行推理
	inputs = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt")
	inputs = {k:v.to(model.device) for k, v in inputs.items()}

	
	result = tokenizer.decode(model.generate(**inputs, do_sample=True, max_length=256)[0], skip_special_tokens=True)
	print(result)