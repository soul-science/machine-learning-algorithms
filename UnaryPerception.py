"""
    Module: UnaryPerception
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/11/30
    Introduce: This is a machine learning algorithm that implements UnaryPerception
"""
import numpy as np
from random import randrange
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class UnaryPerception(object):
    """
        :keyword
            The class of UnaryPerception

        :arg
            speed: the learning speed of algorithm, default: 0.2
            b: the 'b' is offset of the algorithm , default: 0
            w: the 'w' is features of the data, its default value is 0, dimension as characteristic dimension

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
        self.total_error = None

    def __equation(self, x):
        """
        :param x:
            one of the train feature of all
        :return:
            return 1 or -1 by '__sign' method
        """
        return self.__sign(self.w.dot(x.T) + self.b)

    def __sign(self, f):
        """
        :param f:
            f(x)
        :return:
            1 or -1
        """
        return 1 if f >= 0 else -1

    def __classify(self, x, y):
        """
        :param x:
            train features
        :param y:
            train targets
        :return:
            return the failure to judge the data(features, targets)
        """
        error_x, error_y = np.array([], dtype=float), np.array([], dtype=float)
        for i in range(len(x)):
            if self.__equation(x[i]) != y[i]:
                if error_x != []:
                    error_x = np.vstack((error_x, x[i]))
                else:
                    error_x = np.append(error_x, x[i])
                error_y = np.append(error_y, y[i])
        return error_x, error_y

    def __gradient_descent(self, x, y):
        """
        :param x:
            one of the train features
        :param y:
            the corresponding target value
        :return:
            None
        """
        self.w = self.w + self.speed * y * x
        self.b = self.b + self.speed * y

    '''def __can_classify(self, error_x, error_y):
        func = lambda x, y: -y * (self.w.dot(x.T) + self.b)
        errors = sum([func(error_x[i], error_y[i]) for i in range(len(error_x))])
        print(self.total_error, errors)
        if self.total_error is not None and errors > self.total_error:
            return False
        self.total_error = errors
        return True
        '''

    def __get_accuracy(self, predicts: np.array, test_y: np.array):
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

    def fit(self, train_x: iter, train_y: iter):
        """
        :param train_x:
            the features of train data
        :param train_y:
            the targets of train data
        :return:
            return None
        """
        train_x, train_y = np.array(train_x), np.array(train_y)
        if self.w is None:
            self.w = np.array([1]*len(train_x[0]), dtype=float)
        while True:
            error_x, error_y = self.__classify(train_x, train_y)
            if len(error_x) == 0:
                break
            # if self.__can_classify(error_x, error_y) is False:
            #     raise Exception("The train data is not classify...")
            error = randrange(len(error_x))
            self.__gradient_descent(error_x[error], error_y[error])

    def predict(self, test_x: iter):
        """
        :param test_x:
            the features of test data
        :return:
            return the predicts of test data
        """
        predicts = np.array([], dtype=int)
        for x in test_x:
            predicts = np.append(predicts, self.__equation(x))
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
    iris = load_iris()
    x = iris.data[0:100, :]
    y = iris.target[0:100]
    y[y == 0] = -1
    train_x, test_x, train_y, test_y = train_test_split(x, y)
    pro = UnaryPerception()
    pro.fit(train_x, train_y)
    print("the predict of test data:\n", pro.predict(test_x))
    print("the accuracy of test data:\n", pro.score(test_x, test_y))
