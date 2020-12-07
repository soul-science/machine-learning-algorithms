"""
    Module: NaiveBayes
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/12/7
    Introduce: This is a machine learning algorithm that implements NaiveBayes
"""
import numpy as np


class NaiveBayes(object):
    """
        :keyword
            The class of NaiveBayes

        :arg
           alpha: the Laplace smoothing coefficient, {default: 1.0}

        :steps
            1. prepare data set
            2. use the 'fit' method
            3. you can use 'predict' or 'score' to predict the test data and to get the accuracy of train

        :sample
            the test data:
                [-1  1  1]
            the predict of test data:
                [-1.  1.  1.]
            the accuracy of test data:
                1.0
            time had run:
                0.000997781753540039
            Steps to see the main...
    """
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.yp = {}
        self.features = None
        self.targets = None

    def __get_prior_probability(self, train_x, train_y):
        """
        # TODO: to get the prior probability...
        :param train_x:
            the features of train data
        :param train_y:
            the targets of train data
        :return:
            return None
        """
        xp = {}
        set_y = np.array(list(set(train_y)))
        self.targets = set_y
        set_y.shape = (set_y.shape[0], 1)
        t = (train_y == set_y)
        for i in range(t.shape[0]):
            self.yp[set_y[i][0]] = sum(t[i]) / train_y.shape[0]

        set_x = np.apply_along_axis(lambda x: np.array(list(set(x))), 0, train_x)
        set_x = set_x.T
        self.features = set_x

        for i in range(train_x.shape[0]):
            for j in range(train_x.shape[1]):
                xp.setdefault(((j, train_x[i][j]), train_y[i]), 0)
                xp[((j, train_x[i][j]), train_y[i])] += 1
        for i in xp.keys():
            xp[i] = (xp[i] + self.alpha) / (train_x.shape[0] + train_x.shape[1] * self.alpha)

        for i in range(set_y.shape[0]):
            for j in range(set_x.shape[0]):
                for k in range(set_x.shape[1]):
                    self.yp[((j, set_x[j][k]), set_y[i][0])] = \
                        xp[((j, set_x[j][k]), set_y[i][0])] / self.yp[set_y[i][0]]

    def __mul(self, array, y):
        """
        :param array:
            one array of a test data
        :param y:
            one target of all targets
        :return:
            return the y's posterior probability
        """
        s = 1
        for cell in range(self.features.shape[0]):
            s *= self.yp[((cell, array[cell]), y[0])]

        return s * self.yp[y[0]]

    def __predict(self, test_x):
        """
        :param test_x:
            the features of test data
        :return:
            return predicts of the test data
        """
        predicts = np.array([])
        for x in test_x:
            every = np.array([])
            for y in self.targets:
                every = np.append(every, self.__mul(x, y))
            predicts = np.append(predicts, self.targets[np.argmax(every)])

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
        self.__get_prior_probability(train_x, train_y)

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
    t = time.time()
    bayes = NaiveBayes()
    train_x = np.array([[1, 1], [1, 2], [1, 2], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 3], [2, 3],
                        [3, 3], [3, 2], [3, 2], [3, 3], [3, 3]])
    train_y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    test_x = np.array([[2, 1], [1, 3], [2, 2]])
    test_y = np.array([-1, 1, 1])

    bayes.fit(train_x, train_y)

    print("the test data:\n", test_y)
    print("the predict of test data:\n", bayes.predict(test_x))
    print("the accuracy of test data:\n", bayes.score(test_x, test_y))
    print("time had run:\n", time.time() - t)


