"""
tf demo：使用linear regression /线性回归 拟合一个数据集

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

if __name__ == '__main__':
    # 创建训练集：x，y
    x_data = np.arange(100, step=.1)
    y_data = x_data + 20 * np.sin(x_data / 10)
    # 画图
    plt.scatter(x_data, y_data)
    plt.title('whole dataset')
    plt.show()

    # 参数大小
    n_samples = 1000  # 总共的训练数据大小，100/0.1=1000
    batch_size = 100
    # 数据转化为矩阵，本来是两个向量，分别变成二维样本数据
    x_data = np.reshape(x_data, (n_samples, 1))
    y_data = np.reshape(y_data, (n_samples, 1))

    # 定义batch的placeholder
    x = tf.placeholder(tf.float32, shape=(batch_size, 1))
    y = tf.placeholder(tf.float32, shape=(batch_size, 1))

    # 定义要学习的目标 w和b
    with tf.variable_scope('LR'):
        w = tf.get_variable('w', (1, 1), initializer=tf.random_normal_initializer())  # 线性回归的w，并随机初始化
        b = tf.get_variable('b', (1,), initializer=tf.constant_initializer(.0))  # 线性回归的b，并初始化0

    # 正向传播，得到输出，计算loss
    y_pred = tf.matmul(x, w) + b
    loss = tf.reduce_sum((y - y_pred) ** 2 / n_samples)  # loss采用预测和训练的y的方差

    # 根据loss进行bp优化
    train_op = tf.train.AdamOptimizer().minimize(loss)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())  # 初始化所有tf变量

        for step in range(500):
            indices = np.random.choice(n_samples, batch_size)  # 从训练集大小中，随机选minibatch的对应的下标
            x_batch, y_batch = x_data[indices], y_data[indices]  # 从训练集大小中，随机选minibatch的对应的数据
            loss_val, _ = sess.run([loss, train_op],
                                   feed_dict={x: x_batch, y: y_batch})  # 运行loss 和 train ，进行正向传播和反向传播

            if step % 100 ==0:
                print("step {}: loss = {}".format(step, loss_val))
        print(sess.run(w))
        print(b.eval())


