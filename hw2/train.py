import os
import torch
from dataloader import BCIDataSet
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from net import EEGNet, DeepConvNet
from config import \
    NETWORK, EPOCH_NUM, CUDA_DEVICE, OPTIMIZER, LEARNING_RATE, LOSS_FUNC, BATCH_SIZE, ACTIVATE_FUNC, WEIGHT_DECAY


def get_checkpoint_path():
    return 'checkpoint/%s/%s/%s/%s/batch%d' % (NETWORK, ACTIVATE_FUNC, LOSS_FUNC, OPTIMIZER, BATCH_SIZE)


def save_network(net, epoch):
    checkpoint_path = get_checkpoint_path()
    os.makedirs(checkpoint_path, exist_ok=True)

    save_model_path = '%s/model_epoch%.3d.pth' % (checkpoint_path, epoch)
    torch.save(net.state_dict(), save_model_path)


def initialize_network():
    network = {
        'EEGNet': EEGNet(),
        'DeepConvNet': DeepConvNet()
    }[NETWORK]
    network = network.to(CUDA_DEVICE)

    loss_func = {
        'L1': L1Loss(),
        'MSE': MSELoss(),
        'CrossEntropy': CrossEntropyLoss()
    }[LOSS_FUNC]

    optimizer = {'SGD': SGD(params=network.parameters(),
                            lr=LEARNING_RATE,
                            momentum=0.9),
                 'Adam': Adam(params=network.parameters(),
                              lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
                 }[OPTIMIZER]

    print(network)
    return network, loss_func, optimizer


def train(dataloader):
    network, loss_func, optimizer = initialize_network()
    network.train()

    epoch_avg_loss = 0.0
    n = 0

    for epoch in range(EPOCH_NUM):
        for data in dataloader:
            # Forward pass inputs to the network
            inputs, labels = data[0].to(CUDA_DEVICE), data[1].to(CUDA_DEVICE)
            optimizer.zero_grad()
            outputs = network(inputs)

            # Calculate the loss between prediction and ground truth
            loss = loss_func(outputs, labels.long().argmax(axis=1))

            # Calculate average loss of one epoch
            epoch_avg_loss += loss.item()
            n += 1

            # Back-propagation and update weight
            loss.backward()
            optimizer.step()

        # Save the weights of network of one epoch
        save_network(network, epoch + 1)

        # Print average loss in one epoch
        epoch_avg_loss /= n
        print('Epoch %d, Loss = %.6f' % (epoch + 1, epoch_avg_loss))
        epoch_avg_loss = 0.0
        n = 0


if __name__ == '__main__':
    dataset = BCIDataSet('train')
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    train(dataloader)
