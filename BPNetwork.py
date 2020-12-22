"""
    Module: BPNetwork
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/12/23
    Introduce: The BP neural network can set the number of hidden layers,
            the number of neurons in each layer and the number of neurons in the output layer
    介绍: 可设置隐藏层的层数和每层神经元的数量、输出层的神经元数量的BP神经网络(实现自动化)
"""

import numpy as np


class BPNetwork(object):
    """
        :keyword
            The class of QuadrantsNeuralNetwork

        :arg
            learningSpeed: The speed of learning, default: 0.001
            penaltyCoefficient: the penalty coefficient of penalty term, default: 0.001

        :steps
            1. prepare data set
            2. use the 'set' method to set the hidden layers and output layer
            3. use the 'fit' method to train the train data
            4. you can use 'predict' or 'score' to predict the test data and to get the accuracy of train

        :sample
            Iris data set in sklearn module:
                the 5000 cache's loss is 0.05643693359469713
                the test y_data:
                    [2 2 2 0 0 0 2 1 2 1 1 1 1 0 0 0 0 1 2 0 2 0 1 0 2 0 2 2 2 0 1 1 2 1 0 1 1
                    2]
                the predict of test_y is :
                    [2 2 2 0 0 0 2 1 2 1 1 1 1 0 0 0 0 1 2 0 2 0 1 0 2 0 2 2 2 0 1 1 2 1 0 1 1
                    2]
                the accuracy of test is:
                    1.0
                time had run:
                    2.5861566066741943
            Steps to see the main...
    """
    def __init__(self, learningSpeed=0.001, penaltyCoefficient=0.001):
        self.learningSpeed = learningSpeed
        self.penaltyCoefficient = penaltyCoefficient
        self.input_dim = None   # input layer's dim
        self.output_dim = None    # output layer's dim
        self.levels = None
        self.hidden_dim = []    # hidden layer's dim
        self.__w = []   # H = XW + b, W
        self.__b = []   # H = XW + b, b
        self.__h = []   # H = XW + b, H
        self.__relu_h = []  # ReLu(H)

    def __initialize(self):
        """
        # TODO: Initialization of H, ReLu(H), W and b
        :return:
            return None
        """
        np.random.seed(1)
        self.__h = [0] * self.levels
        self.__relu_h = [0] * self.levels
        self.__w.append(np.random.randn(self.input_dim, self.hidden_dim[0]))
        self.__b.append(np.zeros((1, self.hidden_dim[0])))
        for i in range(self.levels-1):
            self.__w.append(np.random.randn(self.hidden_dim[i], self.hidden_dim[i+1]))
            self.__b.append(np.zeros((1, self.hidden_dim[i+1])))

        self.__w.append(np.random.randn(self.hidden_dim[self.levels-1], self.output_dim))
        self.__b.append(np.zeros((1, self.output_dim)))

    def __relu(self, h):
        """
        # TODO: the active coating which name is called ReLu
        :param h:
            H = XW + b, H
        :return:
            0 or h
        """
        return np.maximum(0, h)

    def __softmax(self, y):
        """
        # TODO: Normalization of output
        :param y:
            H = XW + b, H  => ReLu(H)
        :return:
            Normalization of output called prob
        """
        prob = np.exp(y - np.max(y, axis=1, keepdims=True))
        return prob / np.sum(prob, axis=1, keepdims=True)

    def __loss(self, prob, train_y):
        """
        # TODO: Calculate the loss called cross entropy
        :param prob:
            the return of softmax layer
        :param train_y:
            the train of data
        :return:
            cross entropy
        """
        return -np.sum(np.log(prob[np.arange(train_y.shape[0]), train_y]) / train_y.shape[0])

    def __gradient_descent(self, level, dw, db):
        """
        # TODO: gradient descent of These parameters
        :param level:
            the level of the net's layer
        :param dw:
            the gradient of w
        :param db:
            the gradient of b
        :return:
        """
        self.__w[level] -= self.learningSpeed * (1 + self.penaltyCoefficient) * dw
        self.__b[level] -= self.learningSpeed * (1 + self.penaltyCoefficient) * db

    def __affine_forward(self, x, w, b):
        """
        # TODO: the forward propagation
        :param x:
            H = XW + b, X
        :param w:
            H = XW + b, W
        :param b:
            H = XW + b, b
        :return:
            return None
        """
        x_row = x.reshape(x.shape[0], -1)
        return np.dot(x_row, w) + b

    def __affine_backward(self, h, x, w):
        """
        # TODO: the back propagation
        :param h:
            H = WX + b, H
        :param x:
            H = WX + b, X
        :param w:
            H = WX + b, W
        :return:
            dx: the gradient of x
            dw: the gradient of w
            db: the gradient of b
        """
        dx = np.dot(h, w.T)
        dx = np.reshape(dx, x.shape)
        x_row = x.reshape(x.shape[0], -1)

        dw = np.dot(x_row.T, h)
        db = np.sum(h, axis=0, keepdims=True)

        return dx, dw, db

    def __predict(self, test_x):
        """
        # TODO: to predict the test data
        :param test_x:
            the features of test data
        :return:
            return the predicts of test data
        """
        predicts = np.array([], dtype=int)
        self.__h[0] = self.__affine_forward(test_x, self.__w[0], self.__b[0])
        self.__relu_h[0] = self.__relu(self.__h[0])
        for j in range(self.levels - 1):
            self.__h[j + 1] = self.__affine_forward(self.__relu_h[j], self.__w[j+1], self.__b[j+1])
            self.__relu_h[j + 1] = self.__relu(self.__h[j + 1])
        Y = self.__affine_forward(self.__relu_h[self.levels-1], self.__w[self.levels], self.__b[self.levels])
        prob = self.__softmax(Y)

        for i in range(test_x.shape[0]):
            predicts = np.append(predicts, np.argmax(prob[i, :]))

        return predicts

    def __get_accuracy(self, predicts, test_y):
        """
        # TODO: to get the accuracy of predict
        :param predicts:
            the predicts of test data
        :param test_y:
            The target value of the test data
        :return:
            return accuracy
        """
        return sum(predicts == test_y) / predicts.shape[0]

    def set(self, levels, hidden_dim, output_dim):
        """
        # TODO: to set the level of hidden layers, the number of neurons per layer(hidden layers, output layer)
        :param levels:
            the level of hidden layers
        :param hidden_dim:
            the number of neurons in each of hidden layers
        :param output_dim:
            the number of neurons in the output layer
        :return:
            return None
        """
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def fit(self, train_x, train_y, repeat=1000):
        """
        # TODO: to train the network
        :param train_x:
            the features of train data
        :param train_y:
            the targets of train data
        :param repeat:
            the degree of train
        :return:
            return None
        """
        self.input_dim = train_x.shape[1]
        self.__initialize()
        for i in range(1, repeat + 1):
            self.__h[0] = self.__affine_forward(train_x, self.__w[0], self.__b[0])
            self.__relu_h[0] = self.__relu(self.__h[0])
            for j in range(self.levels-1):
                self.__h[j+1] = self.__affine_forward(self.__relu_h[j], self.__w[j+1], self.__b[j+1])
                self.__relu_h[j+1] = self.__relu(self.__h[j+1])

            Y = self.__affine_forward(self.__relu_h[self.levels-1], self.__w[self.levels], self.__b[self.levels])
            prob = self.__softmax(Y)
            loss = self.__loss(prob, train_y)

            # 输出(output the loss of caches)
            print("the %s cache's loss is %s" % (i, loss))

            # 反向传播(the back propagation)
            prob[np.arange(train_y.shape[0]), train_y] -= 1

            dh, dw, db = self.__affine_backward(prob, self.__relu_h[self.levels-1], self.__w[self.levels])
            dh[self.__relu_h[self.levels-1] <= 0] = 0
            self.__gradient_descent(self.levels, dw, db)

            for j in range(self.levels-1, 0, -1):
                dh, dw, db = self.__affine_backward(dh, self.__relu_h[j-1], self.__w[j])
                dh[self.__relu_h[j-1] <= 0] = 0
                self.__gradient_descent(j, dw, db)

            dx, dw, db = self.__affine_backward(dh, train_x, self.__w[0])
            self.__gradient_descent(0, dw, db)

    def predict(self, test_x):
        """
        :param test_x:
            the features of test data
        :return:
            return the predicts of test data
        """
        return self.__predict(test_x)

    def score(self, test_x, test_y):
        """
        :param test_x:
            the features of test data
        :param test_y:
            the targets of test data
        :return:
            return accuracy
        """
        predicts = self.__predict(test_x)
        return self.__get_accuracy(predicts, test_y)


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target)

    import time
    t = time.time()
    net = BPNetwork(learningSpeed=0.0001, penaltyCoefficient=0.001)
    net.set(2, [50, 25], 4)
    net.fit(train_x, train_y, 5000)
    print("the test y_data:\n", test_y)
    print("the predict of test_y is :\n", net.predict(test_x))
    print("the accuracy of test is:\n", net.score(test_x, test_y))
    print("time had run:\n", time.time() - t)
