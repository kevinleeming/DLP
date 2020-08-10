import os
import re
import sys
import torch
import numpy as np
from glob import glob
from dataloader import BCIDataSet
from torch.utils.data import DataLoader
from net import EEGNet, DeepConvNet
from config import NETWORK, CUDA_DEVICE, OPTIMIZER, LOSS_FUNC, BATCH_SIZE, ACTIVATE_FUNC


def get_checkpoint_path():
    return 'checkpoint/%s/%s/%s/%s/batch%d' % (NETWORK, ACTIVATE_FUNC, LOSS_FUNC, OPTIMIZER, BATCH_SIZE)


def get_newest_epoch():
    checkpoint_path = get_checkpoint_path()
    model_paths = sorted(glob('./%s/model*' % checkpoint_path))
    newest_model = model_paths[-1]
    epoch_num = int(re.findall(r'epoch([0-9]+)\.pth', newest_model)[0])

    return epoch_num


def load_network(network, epoch):
    checkpoint_path = get_checkpoint_path()
    network_path = '%s/model_epoch%.3d.pth' % (checkpoint_path, epoch)
    if not os.path.exists(network_path):
        raise FileNotFoundError('Cannot find the network: %s' % network_path)

    network.load_state_dict(torch.load(network_path))

    return network


def initialize_network():
    network = {
        'EEGNet': EEGNet(),
        'DeepConvNet': DeepConvNet()
    }[NETWORK]
    network = network.to(CUDA_DEVICE)

    return network


def get_correct_num(predicts, labels):
    correct_arr = torch.argmax(predicts, dim=1) == torch.argmax(labels, dim=1)
    correct_num = correct_arr.sum()

    return int(correct_num)


def test(dataloader, epoch):
    network = initialize_network()
    network = load_network(network, epoch)
    network.eval()

    accuracy = 0.0

    with torch.no_grad():
        for data in dataloader:
            # Forward pass inputs to the network
            inputs, labels = data[0].to(CUDA_DEVICE), data[1].to(CUDA_DEVICE)
            outputs = network(inputs)

            accuracy += get_correct_num(predicts=outputs, labels=labels)

        data_num = len(dataloader.dataset)
        accuracy = accuracy / data_num * 100

        print('Epoch %d, accuracy = %.3f' % (epoch, accuracy) + ' %')

    return accuracy


if __name__ == '__main__':
    epoch = get_newest_epoch() if len(sys.argv) == 1 else int(sys.argv[1])

    print('Testing')
    test_dataset = BCIDataSet('test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    test_accuracy_list = []
    for e in range(1, epoch+1):
        accuracy = test(test_dataloader, e)
        test_accuracy_list.append(accuracy)

    print('Training')
    train_dataset = BCIDataSet('train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    train_accuracy_list = []

    for e in range(1, epoch+1):
        accuracy = test(train_dataloader, e)
        train_accuracy_list.append(accuracy)

    train_accuracy_list = np.array(train_accuracy_list)
    best_train_accuracy = np.max(train_accuracy_list)
    best_train_accuracy_index = np.argmax(train_accuracy_list) + 1

    test_accuracy_list = np.array(test_accuracy_list)
    best_test_accuracy = np.max(test_accuracy_list)
    best_test_accuracy_index = np.argmax(test_accuracy_list) + 1

    print('Best train accuracy: {} (epoch {})'.format(best_train_accuracy, best_train_accuracy_index))
    print('Best test accuracy: {} (epoch {})'.format(best_test_accuracy, best_test_accuracy_index))

    print('Save accuracy to npy file')

    os.makedirs('accuracy_result', exist_ok=True)
    train_file_name = 'accuracy_result/{0}_{1}_train.npy'.format(NETWORK, ACTIVATE_FUNC)
    test_file_name = 'accuracy_result/{0}_{1}_test.npy'.format(NETWORK, ACTIVATE_FUNC)

    np.save(train_file_name, train_accuracy_list)
    np.save(test_file_name, test_accuracy_list)
