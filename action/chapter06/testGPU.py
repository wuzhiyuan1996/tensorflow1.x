
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_dynamic():
    tf.enable_eager_execution()
    x = tf.random.normal(shape=[10, 10])
    x_square = tf.matmul(x, x)
    # print(x_square)



if __name__ == '__main__':
    test_dynamic()