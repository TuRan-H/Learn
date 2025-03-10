import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# # 定义模型

num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

class RNNModel(nn.Module):
	"""循环神经网络模型"""
	def __init__(self, rnn_layer, vocab_size, **kwargs):
		super(RNNModel, self).__init__(**kwargs)
		self.rnn = rnn_layer
		self.vocab_size = vocab_size
		self.num_hiddens = self.rnn.hidden_size
		self.linear = nn.Linear(self.num_hiddens, self.vocab_size)

	def forward(self, inputs, state):
		X = F.one_hot(inputs.T.long(), self.vocab_size)
		X = X.to(torch.float32)
		Y, state = self.rnn(X, state)
		output = self.linear(Y.reshape((-1, Y.shape[-1])))
		return output, state

	def begin_state(self, device, batch_size=1):
		return torch.zeros(
			(
				self.rnn.num_layers,
				batch_size, 
				self.num_hiddens
			),
			device=device
		)

# # 预测

device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)

num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)


