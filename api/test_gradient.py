
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

w = tfe.Variable(3.0)
b = tfe.Variable(0.0)


def compute_loss(inputs, labels):
    preds = tf.multiply(w, inputs) + b
    return tf.losses.mean_squared_error(labels=labels, predictions=preds)


def train():
    x = [1, 2, 3]
    y = [4, 5, 6]
    data_ = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = data_.repeat().batch(2)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    i = 0
    while i < 6:
        with tf.GradientTape() as tape:
            tape.watch([w, b])
            loss = compute_loss(tf.cast(data[0], dtype=tf.float32), tf.cast(data[1], dtype=tf.int32))
        grads = tape.gradient(target=loss, sources=[w, b])
        optimizer.apply_gradients(grads_and_vars=zip(grads, [w, b]))
        print(w)
        print(b)
        i += 1
        data = iterator.get_next()


if __name__ == '__main__':
    train()