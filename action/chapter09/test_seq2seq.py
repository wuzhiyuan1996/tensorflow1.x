
import tensorflow as tf

class DNN_Encoder(tf.keras.Model): # 编码器模型
    def __init__(self, embedding_dim):
        super(DNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(units=embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units=units)
        self.W2 = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, H, S):
        """
        :param H: shape:[batch_size, max_length, output_size of Encoder],memory，对应编码器Encoder的输出
                  output_size = RNNCell的hidden_size
        :param S: shape:[batch_size, state_size of Decoder], query, 对应解码器的state
                  如果RNNCell为GRUCell，则state_size=hidden_size
                  如果为LSTMCell，则state_size=hidden_size * 2 ,(h,c)
        :return:
        """
        S_with_time_axis = tf.expand_dims(input=S, axis=1) # [batch_size, state_size]->[batch_size, 1, state_size]
        score = tf.nn.tanh(self.W1(S_with_time_axis) + self.W2(H)) # [batch_size, max_length, units]
        # attention weights
        e = tf.nn.softmax(self.V(score), axis=1) # [batch_size, max_length, units]-> [batch_size, max_length, 1]
        # [batch_size, max_length, 1]*[batch_size, max_length, output_size]->[batch_size, output_size]
        context = tf.reduce_sum(e * H, axis=1)
        return context, e


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embeddings =tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim) # [vocab_size, embedding_size]
        self.rnn = self.gru(units=units)
        self.fc1 = tf.keras.layers.Dense(units=self.units)
        self.fc2 = tf.keras.layers.Dense(units=vocab_size) # 分类， 类别总数为vocab_size
        self.attention = BahdanauAttention(units=self.units)

    def call(self, x, H, S):
        """
        :param x: 前一时刻的目标值(ground truth)，不是预测值
        :param H: shape:[batch_size, max_length, output_size of Encoder],memory，对应编码器Encoder的输出
                  output_size = RNNCell的hidden_size
        :param S: shape:[batch_size, state_size of Decoder], query, 对应解码器的state
                  如果RNNCell为GRUCell，则state_size=hidden_size
                  如果为LSTMCell，则state_size=hidden_size * 2 ,(h,c)
        :return:
        """
        context, attention_weights = self.attention(H, S)
        x = self.embeddings(inputs=x) # 前一时刻的目标值(ground_truth)对应的embedding, shape:[batch_size,1, embedding_dim]
        # 当前时刻，解码器Decoder的RNNCell的输入为，x与context的拼接,[batch_size, 1, embedding_dim+units]
        x = tf.concat([tf.expand_dims(context, axis=1), x], axis=-1)
        outputs, states = self.rnn(inputs=x) # rnn的h是否会随着每次调用改变 （感觉是会更新的）
        x = self.fc1(inputs=outputs) # [batch_size,1,units]
        x = tf.reshape(x, shape=[-1, x.shape[2]])
        x = self.fc2(x) # [batch_size *1, vocab_size]?

    def gru(self, units):
        if tf.test.is_gpu_available():
            return tf.keras.layers.CuDNNGRU(units=units, return_sequences=True, return_state=True,
                                            recurrent_initializer='glorot_uniform')
        else:
            return tf.keras.layers.GRU(units=units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform', recurrent_activation='sigmoid')

    def reset_state(self, batch_size):
        return tf.zeros(shape=[batch_size, self.units])

