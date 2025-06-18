"""
b站教学视频: 【手把手带你实战HuggingFace Transformers-入门篇】基础组件之Datasets
视频地址: https://www.bilibili.com/video/BV1Ph4y1b76w/?spm_id_from=333.788.top_right_bar_window_history.content.click&vd_source=41721633578b9591ada330add5535721

使用自定义的脚本导入数据集

使用的数据集: cmrc2018 https://huggingface.co/datasets/cmrc2018

"""
from datasets import load_dataset

# 使用自定义脚本导入数据集
dataset = load_dataset(path='LearnHuggingface/ch01_Getting_started/load_script.py', split='train')
print(dataset)