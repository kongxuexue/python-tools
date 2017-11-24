'''
MNIST：使用多层 CNN
    使用多层卷积、池化、dropout、softmax构成模型进行训练，模型可以到0.99的准确率，使用gpu效果非常明显100s与2400s
'''
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time


def weight_variable_init(shape):
    '''
    参数初始化，初始值为截断的正态分布值，避免全是0造成对称性和0梯度问题
    :param shape:
    :return:
    '''
    init_val = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
    # 使用从截断的正态分布中输出随机值。生成的值服从指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
    return tf.Variable(initial_value=init_val)


def bias_variable_init(shape):
    '''
    偏差初始化，初始值均设置为0.1，有利于在使用ReLU时避免节点输出恒为0成为死节点（dead neurons）
    :param shape:
    :return:
    '''
    init_val = tf.constant(0.1, shape=shape)
    return tf.Variable(init_val)


if __name__ == '__main__':
    mnist = input_data.read_data_sets(train_dir='./MNIST_data', one_hot=1)

    x_in = tf.placeholder(tf.float32, shape=[None, 28 * 28])

    # 每一层都可以创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀，比如b_conv1会变成'hidden1/b_conv1'
    with tf.name_scope('hidden1') as scope:
        # 第一层网络：cnn卷积、max pooling池化
        layer1_intro = '''
            第一层卷积使用5*5的Patch/Kernel/Filter
            卷积得到的宽度为：（含padding后的数据宽度-patch宽度）/stride步长+1，一般需要设置stride为可整除,patch一般较小
            对28*28数据和5*5的patch，padding左右都是2个宽度共28+2+2=32，卷积后得到的是（32-5）/1+1=28*28的数据
        '''
        W_conv1 = weight_variable_init(shape=[5, 5, 1, 32])  # 卷积的权重张量，最后的32是out_channels是自己设定的，假设提取出32个特征通道
        b_conv1 = bias_variable_init(shape=[32])  # 卷积的偏差
        # 为了使用卷积，先将输入x转换成多个28，28，通道数为1的数据，如果是rgb3通道就应该是3
        x_image = tf.reshape(x_in, [-1, 28, 28, 1])
        # 将图片进行same padding，第2,3维设置步长为1，进行 relu(x·w + b)
        hidden_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, filter=W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        # 池化，使用max pooling，设置k是2*2大小，步长是2*2，feature map是14*14
        hidden_pool1 = tf.nn.max_pool(hidden_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        param_intro = '''
            filter：[filter_height, filter_width, in_channels, out_channels]
            strides：2d卷积必须strides[0]=strides[3]=1，第2,3维对应的是图像的高度、宽度上的步长
            padding：'VALID'表示不进行padding，'SAME'表示尝试进行zero padding，patch-1是偶数如4就左右各一半，是奇数如5就左奇右偶
        '''

    with tf.name_scope('hidden2') as scope:
        # 第二层网络：同第一层网络
        W_conv2 = weight_variable_init(shape=[5, 5, 32, 64])  # 卷积的权重张量，in_channels是上一层out_channels，设定本层的out_channels=64
        b_conv2 = bias_variable_init(shape=[64])  # 卷积的偏差
        # 对上一层14*14和本层patch5*5，左右各padding 2得到18*18，卷积结果宽度为（18-5）/1+1=14
        hidden_conv2 = tf.nn.relu(tf.nn.conv2d(hidden_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        # 14*14通过2*2的max pooling得到7*7，通道数为64
        hidden_pool2 = tf.nn.max_pool(hidden_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('hidden3') as scope:
        # 第三层网络；fully connected fc全连接层/dense层/密集连接层、dropout
        '''
            将上一层64通道7*7的图片转变为行向量，与设定个数的全连接层相连，通过relu(wx+b)
        '''
        W_fc3 = weight_variable_init([7 * 7 * 64, 1024])  # 输入是64通道，7*7的图片，假设有1024个全连接层神经元
        b_fc3 = bias_variable_init([1024])
        hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7 * 7 * 64])  # 将上一层张量reshape成多张行向量，和fc层全连接
        hidden_fc3 = tf.nn.relu(tf.matmul(hidden_pool2_flat, W_fc3) + b_fc3)  # relu(wx+b)
        keep_probability = tf.placeholder("float")  # 传入dropout的概率
        hidden_fc3_dropout = tf.nn.dropout(hidden_fc3,
                                           keep_prob=keep_probability)  # 使用dropout减少过拟合，使用feed在训练时使用dropout，测试时关闭
        '''
            dropout除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。???
        '''

    with tf.name_scope('output') as scope:
        # 输出层：全连接层链接到10个输出神经元，使用softmax回归进行分类
        W_out4 = weight_variable_init([1024, 10])
        b_out4 = bias_variable_init([10])
        y_out = tf.nn.softmax(tf.matmul(hidden_fc3_dropout, W_out4) + b_out4)

    # 评估与训练
    '''
        使用交叉熵loss，使用Adam梯度下降
    '''
    y_true = tf.placeholder(tf.float32, [None, 10])  # 数据结果的真实分布
    cross_entropy = -tf.reduce_sum(y_true * tf.log(y_out))  # 交叉熵
    train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(x=tf.argmax(y_out, 1), y=tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    begin = time.time()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())  # init op

        for i in range(0, 20001):
            batch = mnist.train.next_batch(50)
            session.run(train_op,
                        feed_dict={x_in: batch[0], y_true: batch[1], keep_probability: 0.5})  # 每次训练模型，并开启dropout
            if i % 100 == 0:
                # 每100次测试准确率，关闭dropout
                train_accuracy = session.run(accuracy,
                                             feed_dict={x_in: batch[0], y_true: batch[1], keep_probability: 1.0})
                print('step {}, training accuracy: {:.5f}'.format(i, train_accuracy))
        print('train over! test accuracy: {:.5f}'.format(
            session.run(accuracy, feed_dict={x_in: mnist.test.images, y_true: mnist.test.labels, keep_probability: 1.0})))

    end = time.time()
    print('total time:{:.2f}s'.format(end - begin))
