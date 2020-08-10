import torch
import numpy as np
from torch.utils.data import Dataset


def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    tmp_train_label = np.zeros((len(train_label), int(np.max(train_label)) + 1))
    tmp_train_label[:, 1] = train_label
    tmp_train_label[:, 0] = 1 * (train_label == 0)
    train_label = tmp_train_label

    tmp_test_label = np.zeros((len(test_label), int(np.max(test_label)) + 1))
    tmp_test_label[:, 1] = test_label
    tmp_test_label[:, 0] = 1 * (test_label == 0)
    test_label = tmp_test_label

    return train_data, train_label, test_data, test_label


class BCIDataSet(Dataset):
    def __init__(self, dataset_type: str):
        assert dataset_type == 'train' or dataset_type == 'test'
        self.dataset_type = dataset_type

        self.train_data, self.train_label, self.test_data, self.test_label = read_bci_data()

    def __len__(self):
        return self.train_data.shape[0] if self.dataset_type == 'train' else self.test_data.shape[0]

    def __getitem__(self, item):
        data = self.train_data[item] if self.dataset_type == 'train' else self.test_data[item]
        label = self.train_label[item] if self.dataset_type == 'train' else self.test_label[item]

        return torch.from_numpy(data).float(), torch.from_numpy(label).float()
