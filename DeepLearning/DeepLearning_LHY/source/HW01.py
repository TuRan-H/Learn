import numpy as np
import torch.nn
from torch.utils.data import Dataset, DataLoader
import csv
import sys, os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


class MyDataSet(Dataset):
    def __init__(self, path, model, target_only=False):
        super(MyDataSet, self).__init__()
        self.model = model
        self.path = path
        self.target_only = target_only
