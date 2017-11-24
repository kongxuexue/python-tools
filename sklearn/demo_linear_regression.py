# encoding=utf-8
'''
参考文章：
    [sklearn.linear_model.LinearRegression手册](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    [sklearn学习笔记之简单线性回归](http://www.cnblogs.com/magle/p/5881170.html)
'''
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()#是否存在截距，默认存在

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
#输出训练数据
print('x-train:',diabetes_X_train.shape)
print('y-train:',diabetes_y_train)
print('train-predict:',regr.predict(diabetes_X_train))

# The coefficients
print('Coefficients: \n', regr.coef_)#coef_存放相关系数，intercept_则存放截距
# The mean squared error
print("Mean squared error: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()