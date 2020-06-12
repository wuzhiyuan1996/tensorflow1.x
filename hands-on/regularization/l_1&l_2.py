
import tensorflow as tf
# import tensorflow.contrib.framework.arg_scope as arg_scope // No module named 'tensorflow.contrib.framework.arg_scope'
from sklearn.datasets import load_breast_cancer

def get_data():
    data = load_breast_cancer()
    print(data.data)
    print(data.target)

def l_x_regularization(inputs, labels, n_input, n_hiddens=[16, 8], n_output=2, activation=tf.nn.sigmoid, scale=0.01, mode="l_1"):
    initializer = tf.variance_scaling_initializer()
    for i, n_hidden in enumerate(n_hiddens):
        weights_init = initializer(shape=[n_input, n_hidden], dtype=tf.float32)
        weights = tf.Variable(initial_value=weights_init, dtype=tf.float32, name="weights_"+str(i))
        biases = tf.Variable(initial_value=tf.zeros(n_hidden), name="biases_"+str(i))
        hidden = activation(tf.matmul(inputs, weights)+biases)
        inputs = hidden
        n_input = n_hidden
    weights_init = initializer(shape=[n_input, n_output], dtype=tf.float32)
    weights = tf.Variable(initial_value=weights_init, dtype=tf.float32, name="weights_")
    biases = tf.Variable(initial_value=tf.zeros(n_output), name="biases_")
    outputs = tf.matmul(tf.matmul(inputs, weights)+biases)
    base_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(indices=labels, depth=n_output, on_value=1, off_value=0)
        , logits=outputs)
    reg_loss = 0.0
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in trainable_variables:
        name = var.split("/")[-1]
        if "weights" in name or "biases" in name:
            reg_loss += tf.reduce_sum(tf.abs(var))
    loss = tf.add(base_loss, scale*reg_loss, name="loss")
    return loss

def l_x_regularization_api(intputs, labels, hiddens=[16, 8], n_output=2, activation=tf.nn.sigmoid, scale=0.01):
    # with arg_scope([tf.layers.dense], kernel_initializer=tf.variance_scaling_initializer(),
    #                kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=scale),
    #                bias_regularizer=tf.contrib.layers.l1_regularizer(scale=scale)):
    #     for n_hidden in hiddens:
    #         inputs = tf.layers.dense(intputs, n_hidden, activation=activation)
    #     outputs = tf.layers.dense(intputs, n_output)
    for n_hidden in hiddens:
        inputs = tf.layers.dense(intputs, n_hidden, activation=activation, kernel_initializer=tf.variance_scaling_initializer(),
               kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=scale),
               bias_regularizer=tf.contrib.layers.l1_regularizer(scale=scale))
    outputs = tf.layers.dense(intputs, n_output)
    base_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(indices=labels, depth=n_output, on_value=1, off_value=0)
        , logits=outputs)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss]+reg_losses, name="loss")
    return loss

if __name__ == '__main__':
    get_data()