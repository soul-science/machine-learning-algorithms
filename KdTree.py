"""
    Module: KdTree
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/12/3
    Introduce: This is a binary tree in K-dimensional space
"""
import numpy as np

from sklearn.datasets import load_iris


class KNode(object):
    """
        :keyword
            A K-dimensional space
        :arg
            data: K-dimensional data
            parent: father node
            left: Left the child
            right: Right the child
            row: The dimensions compared to the current K-dimension node
            pos: The location of the current node distribution in the tree array

    """
    def __init__(self, data=None, parent=None, left=None, right=None, row=None, pos=None):
        self.data = data
        self.parent = parent
        self.left = left
        self.right = right
        self.row = row
        self.pos = pos


class KdTree(object):
    """
        # TODO: Generally used for KNN algorithm
        :keyword
            The binary tree in K-dimensional space
        :arg
            k: The dimension of a dataset's characteristics
            root: Binary tree roots
            current: Current node
            path: A tree array that records whether a node has passed
            mini: The closest nodes

        :steps
            1. Prepare data set
            2. Initialize the class
            3. Use the 'search' method to get the closet n nodes

        :sample
            The data being searched:
                [5.9 3.  5.1 1.8]
            n recent data searched:
                [[0.282842712474618 array([6.1, 3. , 4.9, 1.8])]
                [0.33166247903553997 array([5.8, 2.7, 5.1, 1.9])]
                [0.33166247903553997 array([5.8, 2.7, 5.1, 1.9])]]
            run time:
                0.0029914379119873047

    """
    def __init__(self, data=None):
        self.k = data.shape[1]
        self.root = KNode(np.array(data, dtype=float))
        self.current = self.root
        self.path = [0]*data.shape[0]
        self.mini = None
        self.initialize()

    def __initialize(self, current, depth, p=0):
        """
        :param current:
            Current node
        :param depth:
            Depth of the current node
        :param p:
            Depth of the current node(None --> 0, left-child --> 1, right-child --> 2)
        :return:
            return None
        """
        if len(current.data) == 0:
            return None
        else:
            row = depth % self.k
            current.data = current.data[current.data[:, row].argsort()]
            index = current.data.shape[0] // 2
            current.left = KNode(current.data[: index], current)
            current.right = KNode(current.data[index+1 :], current)
            current.data = current.data[index]
            current.row = row
            if p == 0:
                current.pos = index
            elif p == 1:
                current.pos = current.parent.pos - index
            else:
                current.pos = current.parent.pos + index
            self.__initialize(current.left, depth + 1, p=1)
            self.__initialize(current.right, depth + 1, p=2)

    def __dfs(self, current, data, n):
        """
        :param current:
            Current node
        :param data:
            The data being searched
        :param n:
            The nearest n nodes
        :return:
            return None
        """
        if current is None:
            return

        if self.mini.size == 0:

            if current.data.size != 0:
                self.mini = np.append(self.mini, [np.sqrt(sum((data - current.data)**2)), current.data], axis=0)
                self.mini.shape = (1, 2)
        else:
            dis = np.sqrt(sum((data - current.data) ** 2))

            if self.path[current.pos] <= 1:
                self.mini = np.vstack((self.mini, [dis, current.data]))
                self.mini = self.mini[self.mini[:, 0].argsort()]
                if self.mini.shape[0] > n:
                    self.mini = self.mini[: -1]

            if current.left is not None and current.left.pos is not None and self.path[current.left.pos] == 0:
                if self.mini[-1, 0] > abs(self.mini[-1, 1][current.left.row] - current.data[current.left.row]):
                    self.path[current.left.pos] = 1
                    self.__dfs(current.left, data, n)
            if current.right is not None and current.right.pos is not None and self.path[current.right.pos] == 0:
                if self.mini[-1, 0] > abs(self.mini[-1, 1][current.right.row] - current.data[current.right.row]):
                    self.path[current.right.pos] = 1
                    self.__dfs(current.right, data, n)

        if current.parent is not None:
            self.path[current.parent.pos] += 1
        self.__dfs(current.parent, data, n)

    def initialize(self):
        """
        :return:
            return None
        """
        self.__initialize(self.root, 0)

    def search(self, data, n=1):
        """
        :param data:
            The data being searched
        :param n:
            The nearest n nodes
        :return:
            return The nearest n nodes ---> mini
        """
        self.current = self.root
        data = np.array(data)
        depth = 0
        self.mini = np.array([], dtype=float)
        while len(self.current.data) != 0:
            row = depth % self.k
            if data[row] > self.current.data[row]:
                self.current = self.current.right
            else:
                self.current = self.current.left
            depth += 1

        self.__dfs(self.current, data, n)
        return self.mini


if __name__ == '__main__':
    import time
    iris = load_iris()
    tt = time.time()
    t = KdTree(iris.data[:-1])
    print("The data being searched:\n", iris.data[-1])
    print("n recent data searched:\n", t.search(iris.data[-1], 3))
    print("run time:\n", time.time() - tt)
