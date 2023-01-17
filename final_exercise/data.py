import numpy as np
import torch
from torch.utils.data import Dataset


class corruptData(Dataset):
    def __init__(self, train):
        if train:
            train_list = []
            for i in range(5):
                train_list.append(np.load(f'../data/corruptmnist/train_{i}.npz', allow_pickle=True))
            data = torch.tensor(np.concatenate([t['images'] for t in train_list])).reshape(-1, 1, 28, 28)
            labels = torch.tensor(np.concatenate([t['labels'] for t in train_list]))
        else:
            test_list = np.load(f'../data/corruptmnist/test.npz', allow_pickle=True)
            data = torch.tensor(test_list['images']).reshape(-1, 1, 28, 28)
            labels = torch.tensor(test_list['labels'])

        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index].float(), self.labels[index]

    def __len__(self):
        return self.labels.numel()