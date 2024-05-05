"""
b站教学视频: 【手把手带你实战HuggingFace Transformers-入门篇】基础组件之Datasets
视频地址: https://www.bilibili.com/video/BV1Ph4y1b76w/?spm_id_from=333.788.top_right_bar_window_history.content.click&vd_source=41721633578b9591ada330add5535721

自定义的脚本
"""
import json
import datasets
from datasets import DownloadManager, DatasetInfo

class CMRC2018(datasets.GeneratorBasedBuilder):
	"""
	创建数据类, 这个类必须要继承于 `datasets.GeneratorBasedBuilder`
	"""
	def _info(self) -> DatasetInfo:
		"""
		返回数据集信息, 以 `DatasetInfo` 对象形式返回
		DatasetInfo需要接受两个参数
			1. description: 对这个数据集的描述
			2. features: 数据集中存在什么feature, features需要以 `datasets.Features` 类定义
				`dataset.Features` 接受参数为字典, 
					字典的key是数据集中可能存在的特征 (以 `str` 类型表示),
					字典的value用datasets提供的数据类表示, 比如说 `datasets.Value`
		"""
		return datasets.DatasetInfo(
			description="CMRC2018 trial",
			features=datasets.Features({
				"id": datasets.Value("string"),
				"context": datasets.Value("string"),
				"question": datasets.Value("string"),
				"answers": datasets.features.Sequence({
					"text":  datasets.Value("string"),
					"answer_start": datasets.Value("int32")
				})
			})
		)
	
	def _split_generators(self, dl_manager: DownloadManager):
		"""
		返回一个生成器列表, 每个生成器对应了一个数据集的特定split
		用于生成数据集不同的划分

		比如说我的数据集存在两个split: train和validation
		那么我就需要以列表形式返回两个生成器
		"""
		return [
			# 这里返回 split = train 的数据集, 其中需要给出数据集地址 `filepath`
			datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"file_path": "./dataset/cmrc2018/cmrc2018_trial.json"})
		]
	
	def _generate_examples(self, file_path):
		"""
		给定数据集地址, 读取数据集构造一个生成器
		生成器每次返回数据集中的一个数据项

		注意: 生成器返回数据项的时候, 数据格式需要和 `_info()` 的数据格式完全对应, 并且要和id一起与元组形式返回
		"""
		# 导入文件, 并且构造数据项
		with open(file_path, encoding="utf-8") as f:
			data = json.load(f)
			for example in data["data"]:
				for paragraph in example["paragraphs"]:
					context = paragraph["context"].strip()
					for qa in paragraph["qas"]:
						question = qa["question"].strip()
						id_ = qa["id"]

						answer_starts = [answer["answer_start"] for answer in qa["answers"]]
						answers = [answer["text"].strip() for answer in qa["answers"]]

						# ! yield数据的时候, 需要将id和数据项构造成一个元组返回
						yield id_, {
							"context": context,
							"question": question,
							"id": id_,
							"answers": {
								"answer_start": answer_starts,
								"text": answers,
							},
						}

