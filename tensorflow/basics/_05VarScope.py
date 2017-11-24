"""
tf 使用variable scope 为变量提供命名空间，可以有效避免变量覆盖
"""

import tensorflow as tf
print(__doc__)

if __name__ == '__main__':
    sess = tf.Session()

    with tf.variable_scope("foo"):
        with tf.variable_scope("bar"):
            v = tf.get_variable("v",[3,2])

    assert v.name == "foo/bar/v:0"
    '''
    默认情况下，reuse是false的，获取变量会创建新的变量
    '''
    with tf.variable_scope("foo"):
        v = tf.get_variable("v",[1])
    print(v.name)  # 变量已经存放在tf变量空间中，v算是对实际变量经过tf封装之后的引用，tf变量空间中的变量是不能重名的

    '''
    使用reuse_variables()表明下面get的是tf中已经存在的变量，否则get会导致tf命名空间重复错误。
    但是如果使用了reuse_variables()，但是却找不到对应的空间变量，那么会导致变量找不到错误。
    '''
    with tf.variable_scope("foo"):
        # RNN中经常需要使用reuse变量，因为是同一个cell时空展开成为多个cell
        tf.get_variable_scope().reuse_variables()  # 如果不重用变量，用get变量会尝试创建新的foo/v，出现foo/v变量已存在错误
        v1 = tf.get_variable("v",[1])
    print(v1.name)
    # 也可以使用variable_scope()的reuse参数表明下面是重用空间变量，注意保证变量名foo/v是存在的
    with tf.variable_scope("foo",reuse=True):
        v2 = tf.get_variable("v",[1])
    print(v2.name)

