from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import joblib

import pandas as pd


boston = load_boston()
standscaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(standscaler.fit_transform(boston.data), boston.target)


# x_train = standscaler.fit_transform(x_train)
# x_test = standscaler.fit_transform(x_test)


def boston_my_ridge():
    estimator = joblib.load('./boston_ridge.pkl')
    data = standscaler.fit_transform(boston.data)
    y_predict = estimator.predict(data)

    print("岭回归(保存)(L2正则化惩罚项)真实值与预测值比较：\n", pd.DataFrame({'y_test': boston.target, 'y_predict': y_predict}))
    print("岭回归(保存)(L2正则化惩罚项)均方误差：\n", mean_squared_error(boston.target, y_predict))
    print("岭回归(保存)(L2正则化惩罚项)回归参数：\n", estimator.coef_)


def boston_ridge():
    estimator = Ridge()

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    joblib.dump(estimator, './boston_ridge.pkl')

    print("岭回归(L2正则化惩罚项)真实值与预测值比较：\n", pd.DataFrame({'y_test': y_test, 'y_predict': y_predict}))
    print("岭回归(L2正则化惩罚项)均方误差：\n", mean_squared_error(y_test, y_predict))
    print("岭回归(L2正则化惩罚项)回归参数：\n", estimator.coef_)


def boston_sgd():
    estimator = SGDRegressor()

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)

    print("随机梯度下降优化回归真实值与预测值比较：\n", pd.DataFrame({'y_test': y_test, 'y_predict': y_predict}))
    print("随机梯度下降优化回归均方误差：\n", mean_squared_error(y_test, y_predict))
    print("随机梯度下降优化回归(L2正则化惩罚项)回归参数：\n", estimator.coef_)


def boston_lr():
    estimator = LinearRegression()

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)

    print("正规方程优化回归真实值与预测值比较：\n", pd.DataFrame({'y_test': y_test, 'y_predict': y_predict}))
    print("正规方程优化回归均方误差：\n", mean_squared_error(y_test, y_predict))
    print("正规方程优化回归(L2正则化惩罚项)回归参数：\n", estimator.coef_)


if __name__ == '__main__':
    boston_my_ridge()
    boston_lr()
    boston_sgd()
    boston_ridge()
