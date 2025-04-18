"""
自己实现load_data_time_machine函数, 用以取代d2l.load_data_time_machine函数
主要原因还是d2l的load_data_time_machine函数不能指定默认下载位置
"""

import re
from d2l import torch as d2l


d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a',
)


def read_time_machine():  # @save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine', cache_dir="./dataset"), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corpus_time_machine(max_tokens=-1):
    line = read_time_machine()
    tokens = d2l.tokenize(line, 'char')
    vocab = d2l.Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab


class SeqDataLoader:
    """An iterator to load sequence data."""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """Defined in :numref:`sec_language_model`"""
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset.

    Defined in :numref:`sec_language_model`"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


if __name__ == "__main__":
    train_iter, vocab = load_data_time_machine(32, 10)
    print()
