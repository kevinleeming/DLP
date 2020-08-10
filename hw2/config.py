import os


# Cuda
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
CUDA_DEVICE = 'cuda'  # cuda, cpu

# Network
NETWORK = 'EEGNet'  # EEGNet, DeepConvNet
ACTIVATE_FUNC = 'ReLU'  # ReLU, LeakyReLU, ELU
BATCH_SIZE = 64
OPTIMIZER = 'Adam'  # Adam, SGD
LOSS_FUNC = 'CrossEntropy'  # L1, MSE, CrossEntropy
LEARNING_RATE = 0.002
EPOCH_NUM = 300
WEIGHT_DECAY = 0.01
