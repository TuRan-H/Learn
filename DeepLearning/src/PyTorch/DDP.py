import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    @property
    def device(self):
        return next(self.parameters()).device


@dataclass
class Arguments:
    n_gpu: int = 2
    data_dir: str = "./data"
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 5


def get_datasets(args: Arguments):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


def get_model(args: Arguments):
    model = MLP()

    if args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids=[int(i) for i in range(args.n_gpu)])

    model = model.to(torch.device("cuda:0"))
    return model


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # 切换到训练模式

    bar = tqdm(total=(len(dataloader)))
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(model.module.device), y.to(model.module.device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            bar.set_description(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        bar.update(1)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # 切换到评估模式
    test_loss, correct = 0, 0

    with torch.no_grad():  # 测试时不需要计算梯度，节省显存
        bar = tqdm(total=(len(dataloader)))
        bar.set_description("Testing")
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            bar.update(1)

    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    args = Arguments()
    train_loader, test_loader = get_datasets(args)
    model = get_model(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for _ in range(args.epochs):
        train(train_loader, model, criterion, optimizer)
        test(test_loader, model, criterion)


if __name__ == "__main__":
    main()
