"""
numpy 与 tensorflow 都可以对矩阵进行运算，
但是tf可以创建tensor，可以自动计算导数，支持GPU
"""
import numpy as np
import tensorflow as tf

print(__doc__)

if __name__ == '__main__':
    a = np.zeros((3, 4))
    print("使用tf.convert_to_tensor() 将一个对象转化为tensor" + '=' * 10)
    ta = tf.convert_to_tensor(a)
    with tf.Session() as sess:
        print(sess.run(ta))

    sess = tf.Session()  # 实例一个tf的session
    a = np.zeros((2, 2));
    b = np.ones((2, 2))
    ta = tf.zeros((2, 2));
    tb = tf.ones((2, 2))
    print("np用sum求和" + '=' * 10)
    print(np.sum(b, axis=1))
    print("tf用reduce_sum求和" + '=' * 10)
    print(tf.reduce_sum(b, axis=1))  # axis只会将矩阵按axis维度进行reduce_sum
    print(sess.run(tf.reduce_sum(b, axis=1)))

    # print("tf用matmul运算矩阵乘积，np用np.dot" + '=' * 10)
    # print(sess.run(tf.add(ta,tb)))
    # print(sess.run(tf.subtract(ta,tb)))
    # print(sess.run(tf.multiply(ta,tb)))
    # print(sess.run(tf.divide(ta,tb)))
    '''
    在tensorflow 1.0版本中，reduction_indices被改为了axis，本质上各种sum、mean等都是降维。
    '''
    print(sess.run(tf.reduce_sum(b, axis=[0, 1])))
    print(sess.run(tf.reduce_sum(b, axis=[1, 0])))
    print("tf用reduce_sum求和" + '=' * 10)
    print(tf.reduce_sum(b, reduction_indices=[1]))  # reduction_indices将矩阵按维度进行reduce_sum压缩为一维
    print(sess.run(tf.reduce_sum(b, reduction_indices=[1])))
    print(sess.run(tf.reduce_sum(b, reduction_indices=[0, 1])))
    print(sess.run(tf.reduce_sum(b, reduction_indices=[1, 0])))


    print("tf用matmul运算矩阵乘积，np用np.dot" + '=' * 10)
    print(a.shape)
    print(ta.get_shape())
    a = np.reshape(a, (1, 4))
    ta = tf.reshape(ta, (1, 4))
    print(a)
    print(sess.run(ta))
    print(np.dot(a, np.reshape(b, (4, 1))))
    print(sess.run(tf.matmul(ta, tf.reshape(tb, (4, 1)))))


