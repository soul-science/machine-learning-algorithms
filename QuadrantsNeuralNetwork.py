"""
    Module: QuadrantsNeuralNetwork
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/12/20
    Introduce: A two-layer artificial neural network used to determine the coordinate system
"""
import numpy as np


class QuadrantsNeuralNetwork(object):
    """
        :keyword
            The class of QuadrantsNeuralNetwork

        :arg
            learningSpeed: The speed of learning, default: 0.001
            penaltyCoefficient: the penalty coefficient of penalty term, default: 0.001

        :steps
            1. prepare data set
            2. use the 'fit' method
            3. you can use 'predict' or 'score' to predict the test data and to get the accuracy of train

        :sample
            the test y_data:
                [0 1 2 3]
            the predict of test_y is :
                [0 1 2 3]
            the accuracy of test is:
                1.0
            time had run:
                0.1216738224029541
        Steps to see the main...
    """
    def __init__(self, learningSpeed=0.001, penaltyCoefficient=0.001):
        self.learningSpeed = learningSpeed
        self.penaltyCoefficient = penaltyCoefficient
        self.input_dim = None   # input layer's dim
        self.num_classes = 4    # output layer's dim
        self.hidden_dim = 50    # hidden layer's dim
        self.__w = []   # H = XW + b, W
        self.__b = []   # H = XW + b, b
        self.__h = []  # H = XW + b, H

    def __initialize(self):
        """
        # TODO: Initialization of W and b
        :return:
            return None
        """
        np.random.seed(1)
        self.__w.append(np.random.randn(self.input_dim, self.hidden_dim))
        self.__w.append(np.random.randn(self.hidden_dim, self.num_classes))
        self.__b.append(np.zeros((1, self.hidden_dim)))
        self.__b.append(np.zeros((1, self.num_classes)))

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
        self.__h.append(np.dot(x_row, w) + b)

    def __affine_backward(self, dout, x, w):
        """
        # TODO: the back propagation
        :param dout:
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
        dx = np.dot(dout, w.T)
        dx = np.reshape(dx, x.shape)
        x_row = x.reshape(x.shape[0], -1)

        dw = np.dot(x_row.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)

        return dx, dw, db

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
        self.__w[level] -= self.learningSpeed*(1 + self.penaltyCoefficient) * dw
        self.__b[level] -= self.learningSpeed*(1 + self.penaltyCoefficient) * db

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

    def __predict(self, test_x):
        """
        # TODO: to predict the test data
        :param test_x:
            the features of test data
        :return:
            return the predicts of test data
        """
        self.__h = []
        predicts = np.array([], dtype=int)
        self.__affine_forward(test_x, self.__w[0], self.__b[0])
        h = self.__relu(self.__h[0])
        self.__affine_forward(h, self.__w[1], self.__b[1])
        prob = self.__softmax(self.__h[1])

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
        for i in range(1, repeat+1):
            self.__h = []
            # 正向传播(the forward propagation)
            self.__affine_forward(train_x, self.__w[0], self.__b[0])
            h = self.__relu(self.__h[0])
            self.__affine_forward(h, self.__w[1], self.__b[1])
            prob = self.__softmax(self.__h[1])
            loss = self.__loss(prob, train_y)

            # 输出(output the loss of caches)
            print("the %s cache's loss is %s" % (i, loss))

            # 反向传播(the back propagation)
            prob[np.arange(train_y.shape[0]), train_y] -= 1
            # prob /= train_y.shape[0]

            dh, dw2, db2 = self.__affine_backward(prob, h, self.__w[1])
            dh[h <= 0] = 0
            dx, dw1, db1 = self.__affine_backward(dh, train_x, self.__w[0])
            self.__gradient_descent(1, dw2, db2)
            self.__gradient_descent(0, dw1, db1)

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
    train_x = np.array([[2, 1], [-1, 1], [-1, -1], [1, -1]])
    train_y = np.array([0, 1, 2, 3])
    test_x = np.array([[1, 2], [-1, 2], [-3, -4], [1, -2]])
    test_y = np.array([0, 1, 2, 3])
    import time
    t = time.time()
    net = QuadrantsNeuralNetwork(0.1)
    net.fit(train_x, train_y, 1000)
    print("the test y_data:\n", test_y)
    print("the predict of test_y is :\n", net.predict(test_x))
    print("the accuracy of test is:\n", net.score(test_x, test_y))
    print("time had run:\n", time.time() - t)
