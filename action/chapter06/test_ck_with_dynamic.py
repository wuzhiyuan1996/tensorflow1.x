
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from action.chapter06.test_ck_with_static import generate_data, gen_datast, get_data, show_data

tf.enable_eager_execution()
# container = tfe.EagerVariableStore() 不存在

weight = tf.Variable(tf.random_normal([1]), name="weight", dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), name="bias", dtype=tf.float32)

# 1. 动态图不支持占位符的定义
# 2. 动态图不能使用优化器的minimize方法，需要使用tfe.implicit_gradients方法计算梯度，然后使用优化器的apply_gradients方法

def linear_regression(X):
    return tf.multiply(X, weight)+b


def compute_loss(f, X, y):
    z = f(X)
    loss = tf.reduce_mean(tf.square(y-z))
    return loss


def train():
    datasize = 100
    train_X, train_y  = generate_data(datasize=datasize)
    dataset = gen_datast(train_X, train_y)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grad_fn = tfe.implicit_gradients(compute_loss)

    global_step = tf.train.get_or_create_global_step()
    iter = 0

    training_epoches = 20
    display_step = 2
    batch_size = 10
    plot_steps = []
    plot_losses = []
    while global_step*batch_size / datasize < training_epoches:
        step = int(global_step * batch_size / datasize)
        X = tf.cast(data['X'], dtype=tf.float32)
        y = tf.cast(data['y'], dtype=tf.float32)
        grads_and_vars = grad_fn(linear_regression, X, y)
        optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)  # compute_cost的输入为linear_regression, data['X'], data['y']
        loss = compute_loss(linear_regression, X, y)
        if step % display_step == 0:
            print("Epoch:", step + 1, "loss=", loss, "weight=", weight, "b=", b)
            if not (loss == "NA"):
                # plot_steps.append(global_step.eval()) 不行
                plot_steps.append(iter)
                plot_losses.append(loss)
        data = iterator.get_next()
        global_step = tf.train.get_or_create_global_step()
        iter += 1
    print("Done!!!")
    print(plot_steps)
    show_data(plot_steps, plot_losses)


def grad(X, y):
    with tf.GradientTape() as tape:
        loss_value = compute_loss(linear_regression, X, y)
    return tape.gradient(target=loss_value, sources=[weight, b]) # target对sources的微分


# def get_cost(X, y):
#     with container.as_default(): # 将动态图使用到的layer包装起来，可以保存变量
#         z = tf.layers.dense(inputs=X, units=1, name="f1")
#     loss = tf.reduce_mean(tf.square(y-z))
#     return loss
#
#
# def grad(inputs, targets):
#     with tf.GradientTape() as tape:
#         loss_value = get_cost(inputs, targets)
#     return tape.gradient(target=loss_value, sources=container.trainable_variables()) # target对sources的微分


def info():
    print("Tensorflow version: {}".format(tf.VERSION))
    print("Eager executation: {}".format(tf.executing_eagerly()))


if __name__ == '__main__':
    train()