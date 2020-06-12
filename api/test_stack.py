
import tensorflow as tf
tf.enable_eager_execution()


def test():
    a = tf.constant([1,2,3, 3])
    b = tf.constant([4, 5, 6, 6])
    c = tf.stack([a, b], axis=0)
    print("c:", c)
    d = tf.unstack(c, axis=0)
    e = tf.unstack(c, axis=1) # num默认为axis的length
    print("d:", d)
    print("e:", e)


if __name__ == '__main__':
    test()
