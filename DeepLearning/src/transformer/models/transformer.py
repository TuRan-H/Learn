import torch

from torch import nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.net = nn.Sequential(
            nn.Linear(self.embedding_dim, 4 * self.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * self.embedding_dim, self.embedding_dim),
            # ! FeedForward在过完两个线性层之后, 需要使用dropout, 防止过拟合
            nn.Dropout(self.dropout_rate),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, attention) -> None:
        """
        Args:
            config (TransformerConfig): 配置参数
            attention: 注意力机制实现, 继承于nn.Module的类
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.attention = attention
        self.ffn = FeedForward(self.embedding_dim, self.dropout_rate)
        self.ln1 = nn.LayerNorm(self.embedding_dim)
        self.ln2 = nn.LayerNorm(self.embedding_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X + self.attention(self.ln1(X))
        X = X + self.ffn(self.ln2(X))

        return X


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout_rate: float,
        vocab_size: int,
        max_seq_length: int,
        nums_layer: int,
        positional_encoding: nn.Module,
        attention: nn.Module,
    ) -> None:
        """
        Args:
            config (TransformerConfig): 配置参数
            positional_embedding: 位置编码实现, 继承于nn.Module的类
            attention: 注意力机制实现, 继承于nn.Module的类
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.nums_layer = nums_layer
        self.dropout_rate = dropout_rate
        self.attention = attention
        self.positional_embedding = positional_encoding

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.decoder_blocks = nn.Sequential(
            *[
                DecoderBlock(
                    embedding_dim=self.embedding_dim,
                    dropout_rate=self.dropout_rate,
                    attention=attention,
                )
                for _ in range(self.nums_layer)
            ]
        )
        self.ln = nn.LayerNorm(self.embedding_dim)
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        # ! tie weight, 即token_embedding和lm_head的权重共享
        # 这里lm_head是embedding_dim -> vocab_size的, 因此其weight的shape为(vocab_size, embedding_dim)
        # 和nn.Embeddging的weight的shape一致
        self.lm_head.weight = self.token_embedding.weight
        self.loss_function = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # ! 在初始化的时候, 不要用标准差为1来初始化, 初始值过大会导致模型不收敛
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, X: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Args:
                X: (batch_size, seq_len)
                targets: (batch_size, seq_len)
        """
        batch_size, seq_length = X.size()
        # X过token embedding, token_embedding -> (batch_size, seq_len, embedding_dim)
        token_embedding = self.token_embedding(X)
        # X过positional embedding, positional_embedding -> (batch_size, seq_len, embedding_dim)
        embedding = self.positional_embedding(token_embedding)
        output = self.decoder_blocks(embedding)
        output = self.ln(output)
        # logits -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(output)

        if targets is None:
            loss = None
        else:
            batch_size, seq_length, vocab_size = logits.size()
            logits = logits.view(batch_size * seq_length, vocab_size)
            targets = targets.view(batch_size * seq_length)
            loss = self.loss_function(logits, targets)

        return logits, loss

    def generate(self, inputs: torch.Tensor, max_new_tokens: int):
        """
        Args:
            idx: (batch_size, seq_length) 经过tokenizer编码后的输入序列
            max_new_tokens: 生成的最大token数量
        #"""
        for _ in range(max_new_tokens):
            # 如果序列的长度大于block_size, 则只取最后block_size个token
            inputs = inputs if inputs.size(1) <= self.max_seq_length else inputs[:, -self.max_seq_length :]

            # 获取模型输出的logits, logits -> (batch_size, vocab_size)
            with torch.no_grad():
                logits, _ = self(inputs)
                logits = logits[:, -1, :]
            # 对模型输出的logits进行softmax
            probs = F.softmax(logits, dim=-1)
            # 从概率分布中采样下一个token, 这里使用的是多项式采样 (根据指定的概率分布进行采样)
            next_token = torch.multinomial(probs, num_samples=1)
            # 附加到原始输入序列的末尾, 作为下一次的输入
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs
