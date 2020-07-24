import numpy as np
import matplotlib.pyplot as plt
from math import log

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

def loss_function(mode, y_pred, y):
    assert mode == 'MSE' or mode == 'L1' or mode == 'Cross_Entropy'
    if mode == 'MSE':
        return (y_pred - y) ** 2, 1 / 2 * (y_pred - y)
    elif mode == 'L1':
        diff = y_pred - y
        return abs(diff), (1 if diff > 0 else -1)
    elif mode == 'Cross_Entropy':
        loss = -y * log(y_pred + 1e-9) - (1 - y) * log(1 - y_pred + 1e-9)
        if y_pred == 0:
            der_loss = (1 - y) / (1 - y_pred)
        elif y_pred == 1:
            der_loss = -y / y_pred
        else:
            der_loss = -y / y_pred + (1 - y) / (1 - y_pred)
        return loss, der_loss


class GenData:
    @staticmethod
    def generate_linear(n = 100):
        pts = np.random.uniform(0, 1, (n, 2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            distance = (pt[0] - pt[1]) / 1.414
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n, 1)

    @staticmethod
    def generate_XOR_easy(n):
        inputs = []
        labels = []
        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)

            if 0.1*i == 0.5:
                continue

            inputs.append([0.1*i, 1-0.1*i])
            labels.append(1)
        return np.array(inputs), np.array(labels).reshape(21, 1)

    @staticmethod
    def fetch_data(mode, n):
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData.generate_linear,
            'XOR': GenData.generate_XOR_easy
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size, num_step=2000, print_interval=100, loss_mode='MSE'):

        self.num_step = num_step
        self.print_interval = print_interval
        self.loss_mode = loss_mode

        # Model parameters initialization
        # Please initiate your network parameters here.
        #
        self.hidden1_weights = np.random.randn(2, hidden_size)
        self.hidden2_weights = np.random.randn(hidden_size, hidden_size)
        self.hidden3_weights = np.random.randn(hidden_size, 1)

    @staticmethod
    def show_result(x, y, pred_y):
        """
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] == pred_y.shape[0]
        """
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Predict Result', fontsize=18)

        for i in range(x.shape[0]):
            if pred_y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')

        plt.show()

    def forward(self, inputs):
        # ...
        z1 = sigmoid(np.dot(inputs, self.hidden1_weights))
        z2 = sigmoid(np.dot(z1, self.hidden2_weights))
        y_pred = sigmoid(np.dot(z2, self.hidden3_weights))

        self.x = inputs
        self.z1 = z1
        self.z2 = z2

        return y_pred

    def backward(self):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        """
        # ...
        delta = []
        delta.append(self.der_loss * derivative_sigmoid(self.output))
        delta.append(np.dot(delta[-1], self.hidden3_weights.T) * derivative_sigmoid(self.z2))
        delta.append(np.dot(delta[-1], self.hidden2_weights.T) * derivative_sigmoid(self.z1))
        delta.reverse()
        # update
        lr = 0.2
        self.hidden1_weights -= lr * np.dot(self.x.T, delta[0])
        self.hidden2_weights -= lr * np.dot(self.z1.T, delta[1])
        self.hidden3_weights -= lr * np.dot(self.z2.T, delta[2])

    def train(self, inputs, labels):
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        accuracy_list, loss_list = [], []
        n = inputs.shape[0]

        for epochs in range(self.num_step):
            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss
                #   3. propagate gradient backward to the front
                self.output = self.forward(inputs[idx:idx + 1, :])
                self.error = self.output - labels[idx:idx + 1, :]
                loss, der_loss = loss_function(self.loss_mode, self.output, labels[idx:idx + 1, :])
                self.der_loss = der_loss
                self.backward()


            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs), end='')
                accuracy, loss = self.test(inputs, labels)
                accuracy_list.append(accuracy)
                loss_list.append(loss)

        print('Training finished! ', end='')
        accuracy, loss = self.test(inputs, labels)
        accuracy_list.append(accuracy)
        loss_list.append(loss)
        # np.save(f'./XOR_{self.loss_mode}', [accuracy_list, loss_list])
        e = []
        a = 0
        it = int((self.num_step/self.print_interval) + 1)
        for i in range(it):
            a += 100
            e.append(a)
        e = np.array(e)
        loss_list = np.array(loss_list)

        plt.plot(e, loss_list.reshape(-1), label = self.loss_mode)
        return e, loss_list, self.loss_mode


    def test(self, inputs, labels):
        n = inputs.shape[0]

        error = 0.0
        loss = 0.0
        for idx in range(n):
            result = self.forward(inputs[idx:idx + 1, :])
            error += abs(result - labels[idx:idx + 1, :])
            loss += loss_function(self.loss_mode, result, labels[idx:idx + 1, :])[0]
        error /= n
        print('accuracy: %.2f' % ((1 - error) * 100) + '% ', end='')
        print('loss: %.2f' % loss)

        return (1 - error) * 100, loss


if __name__ == '__main__':
    data, label = GenData.fetch_data('XOR', 100)

    #-----------------Cross_Entropy--------------------
    net = SimpleNet(100, num_step=2000, loss_mode='Cross_Entropy')
    e1, l1, m1 = net.train(data, label)

    pred_result1 = np.round(net.forward(data))
    net.show_result(data, label, pred_result1)

    """
    #----------------------MSE-------------------------
    net2 = SimpleNet(100, num_step=2000, loss_mode='MSE')
    e2, l2, m2 = net2.train(data, label)

    pred_result2 = np.round(net2.forward(data))

    #----------------------MAE--------------------------
    net3 = SimpleNet(100, num_step=2000, loss_mode='L1')
    e3, l3, m3 = net3.train(data, label)

    pred_result3 = np.round(net3.forward(data))

    plt.figure()
    plt.plot(e1, l1.reshape(-1), label='Cross_Entropy')
    plt.plot(e2, l2.reshape(-1), label='MSE')
    plt.plot(e3, l3.reshape(-1), label='L1')
    plt.legend(loc = 'upper right')
    plt.show()

    net.show_result(data, label, pred_result1)
    net2.show_result(data, label, pred_result2)
    net3.show_result(data, label, pred_result3)
    """