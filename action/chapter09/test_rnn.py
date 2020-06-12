
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
tf.enable_eager_execution()


def test_multinomial():
    # multinomial(logits, num_samples, seed=None, name=None, output_dtype=None)
    # logits:[batch_size, num_classes]，每一行为num_classes个类别未softmax的概率，
    # num_samples表示根据每一行的概率产生多少个样本（根据类别的概率选择类别）
    samples = tf.multinomial([[0.4, 0.6], [0.5, 0.7],[0.2, 0.1],[0.7, 0.8]], 2, seed=40)
    with tf.Session() as sess:
        print(sess.run(samples))


class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        """

        :param vocab_size: 语料库大小
        :param embedding_dim:
        :param units:
        :param batch_size:
        """
        super(Model, self).__init__()
        self.units = units
        self.batch_size = batch_size
        # inputs: (batch_size, input_length)
        # outputs: (batch_size, input_length, output_dim)
        self.embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(
                units=self.units, return_sequences=True, return_state=True,
                recurrent_initializer='glorot_uniform')
        else:
            # return_sequence: 是返回整个output序列还是只返回output序列的最后一个
            # return_state: 为true，则除了返回output外，还会返回最后一个state
            self.gru = tf.keras.layers.GRU(units=self.units, return_sequences=True, return_state=True,
                                           recurrent_initializer='glorot_uniform', recurrent_activation='sigmoid')
        self.fc = tf.keras.layers.Dense(units=vocab_size)

    def call(self, x, hidden):
        x = self.embeddings(x) # [batch_size, input_length, embedding_dim]
        # 注意gru的output_t和s_t是一样的，都是h_t
        # 因此，outputs：[batch_size, max_length, hidden_size], states: [batch_size, hidden_size]
        outputs, states = self.gru(inputs=x, initial_state=hidden)
        outputs = tf.reshape(outputs, shape=[-1, outputs.shape[2]])
        x = self.fc(outputs) # [-1, hidden_size] -> [-1, vocab_size] # 得到每一个词的多项式分布
        return x, states


def compute_loss(ground_truth, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=preds)


def train():
    pass


#     lstm_state_as_tensor_shape = [num_layers, 2, batch_size, hidden_size] # lstm 的state包含h和c
#     initial_state = tf.zeros(lstm_state_as_tensor_shape)
#     unstack_state = tf.unstack(initial_state, axis=0)
#     tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1]) for idx in range(num_layers)])
#     inputs = tf.unstack(inputs, num=num_steps, axis=1) # [batch_size, num_steps, dim]
#     outputs, state_out = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=tuple_state)


if __name__ == '__main__':
    test_multinomial()


