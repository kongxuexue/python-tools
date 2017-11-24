'''
平面拟合：创建y=ax+b的x和y，使用优化器进行逼近设置的a和b
'''
import tensorflow as tf
import numpy as np
print(__doc__)
if __name__ == '__main__':

    # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
    x_data = np.float32(np.random.rand(2, 100))  # 随机输入x
    y_data = np.dot([0.100, 0.200], x_data) + 0.300  # 模型拟合目标是 y=[0.1,0.2]x+0.3

    # 构造一个线性模型
    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))  # 设置y=wx+b中的w和b的初始值
    y = tf.matmul(W, x_data) + b

    # 最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化变量
    # init = tf.initialize_all_variables()  # deprecated
    init = tf.global_variables_initializer()

    # 启动图 (graph)
    sess = tf.Session()
    sess.run(init)

    # 拟合平面
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))

    print('最佳拟合结果 >>> W: [[0.100  0.200]], b: [0.300]')
