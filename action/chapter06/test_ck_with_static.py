
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def generate_data(datasize=100):
    train_X = np.linspace(-1, 1, datasize)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
    return train_X, train_Y


def gen_datast(X, y, batch_size=10):
    dataset = tf.data.Dataset.from_tensor_slices({
        'X': X,
        'y': y
    })
    return dataset.repeat().batch(batch_size=batch_size)


def get_data(dataset):
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()
    return data


def show_data(X, y):
    plt.plot(X, y, 'r-')
    plt.show()


def regression(datasize=100):
    train_X, train_y  = generate_data(datasize)
    dataset = gen_datast(train_X, train_y)
    data = get_data(dataset)
    X = tf.cast(data['X'], dtype=tf.float32)
    y = tf.cast(data['y'], dtype=tf.float32)
    weight = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    z = tf.multiply(X, weight) + b
    global_step = tf.Variable(0, name='global_step', trainable=False)
    loss = tf.reduce_mean(tf.square(z-y))
    ops = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss, global_step=global_step)
    init = tf.global_variables_initializer()
    training_epoches = 30
    display_step = 2
    save_dir = r"E:\code\python\deeplearning\tensorflow1.x\data\log\chapter06"
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
    plot_steps = []
    plot_losses = []
    with tf.Session() as sess:
        sess.run(init)
        kpt = tf.train.latest_checkpoint(save_dir)
        if kpt != None:
            saver.restore(sess, kpt) # 恢复
        batch_size = 10
        while global_step.eval() * batch_size / datasize < training_epoches:
            step = int(global_step.eval() * batch_size / datasize)
            _, loss_ = sess.run([ops, loss])
            if step % display_step == 0:
                print("Epoch:", step+1, "loss=", loss, "weight=", sess.run(weight), "b=", sess.run(b))
                if not (loss == "NA"):
                    plot_steps.append(global_step.eval())
                    plot_losses.append(loss_)
                saver.save(sess, save_dir+os.sep+"linear_model.cpkt", global_step)
        print("Done!!!")
    show_data(plot_steps, plot_losses)


if __name__ == '__main__':
    regression()

