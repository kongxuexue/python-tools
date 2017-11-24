'''
MNIST：简单模型 y=softmax(wx+b) 属于简单的logits（回归）模型
    使用简单的模型 y=softmax(wx+b)可以获得的最终accuracy在0.91左右，算是很差的，好的结果在0.97到0.997
'''

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time

print(__doc__)

if __name__ == '__main__':
    # mnist 数据准备与介绍
    mnist = input_data.read_data_sets(train_dir="MNIST_data/", one_hot=True)  # 会自动下载mnist数据文件夹到当前目录，本身可自动覆盖或者避免下载
    mnist_intro = '''
    mnist有50000训练数据和10000测试数据，每个数据是一幅图，由28*28=784个像素强度在0到1之间的数值构成，并且对应一个标签是one-hot的10维数据，值是0到9
    '''
    print(mnist_intro)

    # softmax 多项式逻辑回归
    softmax_intro = '''
    softmax 是一种常用的多分类方法，对于多个可正可负的数值，全都计算exp，再除以所有的exp之和作为相应的概率，可以保证概率和为1
    '''
    print(softmax_intro)

    # 训练简单的MNIST回归模型：y=softmax(wx+b)，神经网络输入是x，隐含参数是w
    x_in = tf.placeholder(tf.float32, shape=[None, 28 * 28])  # 为了可以运行任意数量的数据，使用占位符
    # feed占位符与变量variable相比的区别在于：variable运行中可修改，feed_dict每次需要run时指定比较固定
    w = tf.Variable(initial_value=tf.zeros(shape=[784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 构造模型，输入是[None, 28*28]的x样本，经过[28*28, 10]的权重矩阵w得到[None, 10]的结果，加上偏置b，经过softmax得到[None, 10]的结果，得到的就是模型的预测
    y_predicate = tf.nn.softmax(logits=tf.matmul(x_in, w) + b)

    # 训练模型，利用交叉熵作为loss函数，再使用梯度下降进行参数迭代bp训练
    cross_entropy_intro = '''
    交叉熵的定义是：-sum_i(yt_i)log(yp_i), 期中yt为实际的概率分布，yp为预测的概率分布，对所有的i，求（实际分布*log(预测分布)），最终取负数
    '''
    print(cross_entropy_intro)

    y_true = tf.placeholder(tf.float32, [None, 10])  # 训练集的真实分布

    cross_entropy = -tf.reduce_sum(y_true * tf.log(y_predicate))  # reduce_sum是求和op，该公式对应交叉熵计算，这里对应的是所有选定数据，不是单个图片，性能更好

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(
        loss=cross_entropy)  # 使用tf的梯度下降，设置学习率和损失函数

    # 初始化模型的变量
    init = tf.global_variables_initializer()

    # 评估模型，定义好正确率等参数，以便在训练过程中查看优化过程
    correct_prediction = tf.equal(tf.argmax(y_predicate, axis=1),
                                  tf.argmax(y_true,
                                            axis=1))  # 比较真实和预测[None, 10]的第二维最大值的索引是否相同，这里的结果是[True,False,False,True...]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 将上面的布尔值转成0,1，然后求[1,0,0,1...]的均值就是准确率

    # mnist.train.

    begin = time.time()
    with tf.Session() as session:
        session.run(init)

        for i in range(0, 10001):
            batch_xtrain, batch_ytrain = mnist.train.next_batch(100)  # 没有直接使用所有的数据进行训练是为了减少计算量，防止数据量溢出等情况，这里每次训练100个图像
            session.run(train_op, feed_dict={x_in: batch_xtrain, y_true: batch_ytrain})
            if i % 20 == 0:
                print(session.run(accuracy, feed_dict={x_in: mnist.test.images, y_true: mnist.test.labels}))
    end = time.time()
    print('total time:{:.2f}s'.format(end - begin))
