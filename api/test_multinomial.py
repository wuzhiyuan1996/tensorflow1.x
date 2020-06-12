
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()


def test(): # 多项式分布，采样
    b = tf.constant(np.random.normal(size=[4,2]))
    # Draws samples from a multinomial distribution.
    # num_samples, 每一行采样num_samples个
    print(tf.multinomial(logits=b, num_samples=1))
    print(tf.multinomial(logits=b, num_samples=2))


if __name__ == '__main__':
    # test()
    print(tf.get_variable("a"))