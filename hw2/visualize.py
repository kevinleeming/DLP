import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    networks = ['EEGNet', 'DeepConvNet']
    activate_funcs = ['ReLU', 'LeakyReLU', 'ELU']

    for network in networks:
        plt.figure()
        plt.title('Activation function comparison(%s)' % network)
        plt.margins(y=0.1, tight=False)
        plt.ylim(45, 105)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')

        for activate_func in activate_funcs:
            train_accuracy_path = 'accuracy_result/%s_%s_train.npy' % (network, activate_func)
            test_accuracy_path = 'accuracy_result/%s_%s_test.npy' % (network, activate_func)

            if not os.path.exists(train_accuracy_path):
                # raise FileNotFoundError('Cannot find accuracy result')
                continue

            train_accuracy_list = np.load(train_accuracy_path)
            test_accuracy_list = np.load(test_accuracy_path)

            plt.plot(train_accuracy_list, label='%s_train' % activate_func)
            plt.plot(test_accuracy_list, label='%s_test' % activate_func)

        plt.legend(loc='lower right')
        plt.savefig('accuracy_result/%s_result.png' % network)
        plt.close()
