"""
    Module: KNN
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/12/3
    Introduce: This is a machine learning algorithm that implements KNN
"""
import numpy as np
from KdTree import KdTree   # KdTree of KdTress module

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class KNN(object):
    """
        :keyword
            The class of UnaryPerception

        :arg
            k_neighbors: The nearest k neighbors

        :steps
            1. prepare data set
            2. use the 'fit' method
            3. you can use 'predict' or 'score' to predict the test data and to get the accuracy of train

        :sample
            Iris data set in sklearn module:
                the test y_data:
                    [1 1 1 0 0 2 0 2 1 1 1 1 0 0 0 2 1 0 0 0 2 2 0 2 2 1 2 2 0 1 1 2 2 2 0 0 1
                    0]
                the predict of test x_data:
                    [1 1 1 0 0 2 0 2 2 1 1 1 0 0 0 1 1 0 0 0 2 2 0 2 2 1 1 2 0 1 1 1 2 2 0 0 1
                    0]
                the accuracy of test data:
                    0.8947368421052632
                time had run:
                    0.012001752853393555
            Steps to see the main...
    """
    def __init__(self, k_neighbors):
        self.k_neighbors = k_neighbors
        self.train_x = None
        self.train_y = None
        self.tree = None

    def __predict(self, test_x):
        """
        :param test_x:
            the features of test data
        :return:
            return predicts of the test data
        """
        predicts = np.array([], dtype=int)
        for x in test_x:
            indexes = np.array([], dtype=int)
            gets = self.tree.search(x, self.k_neighbors)[:, 1]
            for get in gets:
                index = np.where((self.train_x == get).all(1))[0]
                indexes = np.append(indexes, index)
            indexes.shape = (indexes.shape[0], 1)
            pre = self.train_y[indexes]
            pre.shape = (pre.shape[0],)
            predicts = np.append(predicts, np.argmax(np.bincount(pre)))
        return predicts

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
        self.train_x = train_x
        self.train_y = train_y
        self.tree = KdTree(train_x)

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
        predicts = self.predict(test_x)
        return self.__get_accuracy(predicts, test_y)


if __name__ == '__main__':
    import time
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target)
    t = time.time()
    knn = KNN(k_neighbors=3)
    knn.fit(train_x, train_y)
    print("the test data:\n", test_y)
    print("the predict of test data:\n", knn.predict(test_x))
    print("the accuracy of test data:\n", knn.score(test_x, test_y))
    print("time had run:\n", time.time() - t)