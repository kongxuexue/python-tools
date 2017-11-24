"""
tensor and session
    tensor称为张量，对应的是数据。操作operator对tensor进行计算，输出tensor。op的运算需要在tf的session（会话）中进行运行（run）。
    session会话用于运行图表Graph，tf默认的图tf.Graph() 可以满足大部分功能.
    tensor的阶指的是张量的维数，比如3*3的二维矩阵是2阶张量
"""

import tensorflow as tf

print(__doc__)

if __name__ == '__main__':

    matrix1 = tf.constant([[3, 2]])  # 每一个tf的操作都是一个operator（op），而常量op（constant）没有输入tensor，只有输出
    matrix2 = tf.constant([[4], [5]], name="constant_op")  # 每个op操作基本都可以设置name
    product = tf.matmul(matrix1, matrix2)  # tf的矩阵乘法 matrix multiply，结果是numpy的ndarray类型

    sess = tf.Session()  # tf 默认的会话，可以满足大部分需求
    result = sess.run(product)  # op必须使用session进行run，session将会将构造的会话自动进行加载和执行
    print(result)

    # 使用with自动释放资源
    with tf.Session() as session:
        print(session.run(product))
        print('只要session已经被定义过，就可以使用会话.', product.eval())

    sess.close()  # 关闭会话，释放资源

    interactiveSession_intro = '''
    使用交互式TensorFlow：
        session每次需要构造好图之后run，交互式可以使用InteractiveSession类代替Session类，
        使用Tensor.eval()或者Operator.run()代替Session.run()。可以避免使用一个变量来保持会话
    '''
    print(interactiveSession_intro)
    # 首先先打开一个交互式session，之后就可以使用tensor.eval() operator.run() 直接运行
    session = tf.InteractiveSession()
    mata = tf.Variable([1, 2])
    matb = tf.zeros([1, 2], dtype=tf.int32)
    mata.initializer.run()  # 使用tensor的op.run
    sub = tf.subtract(matb, mata)  # 减法从sub变为subtract
    print('直接使用tensor的eval运行：{}'.format(sub.eval()))
    print('也可以使用interactiveSession本身run：{}'.format(session.run(sub)))
