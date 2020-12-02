"""
    Module: DualityPerception
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/12/1
    Introduce: This is a machine learning algorithm that implements DualityPerception
"""

import numpy as np
from random import randrange
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class DualityPerception(object):
    """
            :keyword
                The class of UnaryPerception

            :arg
                speed: the learning speed of algorithm, default: 0.2
                b: the 'b' is offset of the algorithm , default: 0
                w: the 'w' is features of the data, its default value is 0, dimension as characteristic dimension
                gram: Gram matrix which is used to record the inner product of the training features
                train_x: training features
                train_y: training targets

            :steps
                1. prepare data set
                2. use the 'fit' method
                3. you can use 'predict' or 'score' to predict the test data and to get the accuracy of train

            :sample
                Iris data set in sklearn module:
                    predict ---> all true
                    accuracy ---> 1
                Steps to see the main...
        """
    def __init__(self, speed=0.2, b=0):
        self.speed = speed
        self.b = b
        self.w = None
        self.gram = None
        self.train_x = None
        self.train_y = None

    def __sign(self, f):
        """
        :param f:
            f(x)
        :return:
            1 or -1
        """
        f[f >= 0] = 1
        f[f < 0] = -1
        return f == self.train_y

    def __equation(self):
        """
        :param x:
            one of the train feature of all
        :return:
            return 1 or -1 by '__sign' method
        """
        return self.__sign((self.w * self.train_y).dot(self.gram) + self.b)

    def __classify(self):
        """
        # TODO: the method used to determined whether the classification is complete and the parameters are updated
        :return:
            return True or False
        """
        predicts = self.__equation()
        if ((predicts == True).all()):
            return True
        while True:
            error = randrange(len(predicts))
            if predicts[error] == False:
                self.w[error] += self.speed
                self.b += self.speed * self.train_y[error]
                break
        return False

    def __gram_builder(self):
        """
        # TODO: the method used to build the gram matrix
        :return:
            return gram matrix
        """
        return self.train_x.dot(self.train_x.T)

    def __predict(self, x):
        """
        :param x:
            'x' is the testing features
        :return:
            return 1 or -1
        """
        return 1 if (self.w * self.train_y).dot(self.train_x).dot(x) + self.b >= 0 else -1

    def __get_accuracy(self, predicts, test_y):
        """
        :param predicts:
            the predicts of test data
        :param test_y:
            The target value of the test data
        :return:
            return accuracy
        """
        total = len(predicts)
        return sum(predicts == test_y) / total

    def fit(self, train_x, train_y):
        """
        :param train_x:
            the features of train data
        :param train_y:
            the targets of train data
        :return:
            return None
        """
        self.train_x, self.train_y = np.array(train_x), np.array(train_y)

        self.w = np.array([0]*len(self.train_x), dtype=float)
        self.gram = self.__gram_builder()
        while True:
            if self.__classify():
                break

    def predict(self, test_x: iter):
        """
        :param test_x:
            the features of test data
        :return:
            return the predicts of test data
        """
        predicts = np.array([], dtype=int)
        for x in test_x:
            predicts = np.append(predicts, self.__predict(x))

        return predicts

    def score(self, test_x: iter, test_y: iter):
        """
        :param test_x:
            the features of test data
        :param test_y:
            the targets of test data
        :return:
            return accuracy
        """
        return self.__get_accuracy(self.predict(test_x), test_y)


if __name__ == '__main__':
    import time
    iris = load_iris()
    x = iris.data[0:100, :]
    y = iris.target[0:100]
    y[y == 0] = -1
    train_x, test_x, train_y, test_y = train_test_split(x, y)
    t = time.time()
    pro = DualityPerception()
    pro.fit(train_x, train_y)
    print("the predict of test data:\n", pro.predict(test_x))
    print("the accuracy of test data:\n", pro.score(test_x, test_y))
    print("time had run: \n", time.time() - t)
