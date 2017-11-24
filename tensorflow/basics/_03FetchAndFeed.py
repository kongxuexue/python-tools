'''
session.run(self,fetches,feed_dict,options,run_metadata)
fetch: 返回tensor经过op获得的结果叫做fetch，
feed: 临时代替tf图中的tensor称为feed

'''
import tensorflow as tf
print(__doc__)

if __name__ == '__main__':
    in1 = tf.constant(2)
    in2 = tf.constant(3)
    in3 = tf.constant(5)
    mysum = tf.add(in1, in2, name="sum")
    mul = tf.multiply(mysum, in3)

    sess = tf.Session()
    res = sess.run([mul])
    print(res)
    res = sess.run([mysum, mul])  # 调用run执行图时，传入的tensor可以返回结果（fetch）
    res = sess.run(fetches=[mysum, mul])  # 调用run执行图时，传入的tensor可以返回结果（fetch）
    print(res)

    with tf.Session() as session:
        print(mysum.eval())  #  eval属于session.run的语法糖
        print(session.run(mysum))
    # feed 可以先用占位符占用图位置，运行时再传入相应的tensor数值，提高灵活性
    in4 = tf.placeholder(dtype=tf.float32, shape=None, name='in4')
    in5 = tf.placeholder(dtype=tf.float32, shape=None, name='in5')
    out = tf.multiply(in4, in5)
    print(sess.run(out, feed_dict={in4: 4, in5: 7}))
