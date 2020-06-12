
import tensorflow as tf
tf.enable_eager_execution()


def test1():
    W  = tf.Variable(initial_value=tf.random.normal(shape=[4,2]))
    print(W)
    w = tf.gather(W, [3, 2, 1, 0], axis=0)
    print(w)

if __name__ == '__main__':
    test1()
    print(tf.reshape(1, []))