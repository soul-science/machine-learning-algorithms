"""
    Module: mnist_train_netModel
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/12/27
    Introduce: The neural network for training the MNIST data set ,
        which can be able to use an instance to repeat the training.
    介绍: 用于训练mnist数据集的神经网络(能够使用一个实例去重复设置参数训练)
"""
import numpy as np

import saveNet
from BPNetwork import BPNetwork
from sklearn.preprocessing import StandardScaler


def initialize():
    """
    # TODO: initialize the original data
    :return:
        train_x, train_y, test_x, test_y
    """
    scaler = StandardScaler()
    train_x = scaler.fit_transform(np.genfromtxt(
        r'C:\Users\shz\Desktop\datasets\mnist\train_img.csv', dtype=int, delimiter=','
    ))
    train_y = np.genfromtxt(
        r'C:\Users\shz\Desktop\datasets\mnist\train_labels.csv', dtype=int, delimiter=','
    )
    test_x = scaler.fit_transform(np.genfromtxt(
        r'C:\Users\shz\Desktop\datasets\mnist\test_img.csv', dtype=int, delimiter=','
    ))
    test_y = np.genfromtxt(
        r'C:\Users\shz\Desktop\datasets\mnist\test_labels.csv', dtype=int, delimiter=','
    )
    return train_x, train_y, test_x, test_y


class NetModel(object):
    """
        :keyword
            The class of NetModel

        :arg
            None

        :steps
            1. use the 'net_setting' method to set the arguments
            2. use the 'fit' method to train the train data
            3. you can use the 'predict' or 'score' method to predict the test data and to get the accuracy of train
            4. you use the 'save' method to save the net, and you can use 'load_model' from SaveNet to load the net

        :sample
            Steps to see the main...
    """
    train_x, train_y, test_x, test_y = initialize()

    def __init__(self):
        self.learningSpeed = None
        self.penaltyCoefficient = None
        self.activation = None
        self.levels = None
        self.hidden_dim = None
        self.output_dim = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.train_x = NetModel.train_x
        self.train_y = NetModel.train_y
        self.test_x = NetModel.test_x
        self.test_y = NetModel.test_y

    def __update_net(self):
        """
        # TODO: update the net(after 'fit' method running)
        """
        self.net = BPNetwork(self.learningSpeed, self.penaltyCoefficient, self.activation)
        self.net.set(self.levels, self.hidden_dim, self.output_dim)

    def net_setting(self, levels, hidden_dim, output_dim, learningSpeed=0.0000005, penaltyCoefficient=0.0001, activation="relu"):
        """
        # TODO: to set the net as BPNetWork from BPNetWork
        :param levels:
            level of hidden layers
        :param hidden_dim:
            each of hidden layers' dim
        :param output_dim:
            output layers' dim
        :param learningSpeed:
            learning's speed
        :param penaltyCoefficient:
            penalty coefficient
        :param activation:
            activation function's name:('relu', 'sigmoid', 'tanh')
        :return:
            return None
        """
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learningSpeed = learningSpeed
        self.penaltyCoefficient = penaltyCoefficient
        self.activation = activation

    def fit(self, repeat=10):
        """
        # TODO: to train the network
        :param repeat:
            the degree of train
        :return:
            return None
        """
        self.__update_net()
        self.net.fit(self.train_x, self.train_y, repeat)

    def predict(self):
        """
        # TODO: to predict the y of test data
        :return:
            return the predicts of test data
        """
        return self.net.predict(self.test_x)

    def score(self):
        """
        # TODO: to get accuracy of test data
        :return:
            return score
        """
        return self.net.score(self.test_x, self.test_y)

    def save(self, path):
        """
        # TODO: to save the net
        :param path:
            the path of file (including file name)
        :return:
            return None
        """
        saveNet.save_model(self.net, path)


if __name__ == '__main__':
    model = NetModel()
    model.net_setting(2, [300, 150], 10, learningSpeed=0.00001, penaltyCoefficient=0.0001, activation="sigmoid")
    import time
    t = time.time()
    model.fit(repeat=10)
    print("the accuracy of test is:\n", model.score())
    print("time had run:\n", time.time() - t)
