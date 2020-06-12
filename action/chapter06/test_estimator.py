
import tensorflow as tf
import numpy as np
from action.chapter06.test_ck_with_static import generate_data, gen_datast

tf.logging.set_verbosity(tf.logging.INFO) # 设置日志级别，所有级别高于INFO的信息都可以输出

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33) # 防止显存占满，规定每个process只是用1/3
session_config = tf.ConfigProto(gpu_options=gpu_options)
# with tf.Session(config=session_config) as sess:


def generate_data(datasize=100):
    train_X = np.linspace(-1, 1, datasize)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
    return train_X, train_Y


def gen_datast(X, y, batch_size=10):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset.repeat().batch(batch_size=batch_size)


def my_model(features, labels, mode, params):
    W = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    predictions = tf.multiply(W, tf.cast(features, dtype=tf.float32)) + b
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
    loss_out = tf.identity(loss, name='loss_out') # 复制张量loss用于显示，取名为loss_out
    mean_loss = tf.metrics.mean(loss)
    metrics = {'mean_loss':mean_loss}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics) # 额外增加指标mean_loss
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradDAOptimizer(learning_rate=params["learning_rate"], global_step=tf.train.get_or_create_global_step())
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def regression():
    tensors_to_log = {"钩子函数loss输出":"loss_out"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1) #  every_n_iter: 每N步打印张量tensors的值
    warm_start_from = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=r"E:\\code\\python\\deeplearning\\tensorflow1.x\\data\\ck",
    )
    config = tf.estimator.RunConfig(session_config=session_config, save_checkpoints_steps=50)
    estimator = tf.estimator.Estimator(
        model_fn=my_model, model_dir=r"E:\\code\\python\\deeplearning\\tensorflow1.x\\data\\ck",
        params={'learning_rate':0.01}, config=config, warm_start_from=warm_start_from)
    X, y = generate_data()
    estimator.train(input_fn=lambda:gen_datast(X, y), steps=300, hooks=[logging_hook]) # 要求input_fn无参，这里使用匿名函数封装
    tf.logging.info("Done!!!")


def predict():
    config = tf.estimator.RunConfig(session_config=session_config, save_checkpoints_steps=50)
    estimator = tf.estimator.Estimator(
        model_fn=my_model, model_dir=r"E:\\code\\python\\deeplearning\\tensorflow1.x\\data\\ck",
        params={'learning_rate': 0.01}, config=config)
    new_samples = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=new_samples, y=None, batch_size=1, num_epochs=1, shuffle=True)
    predictions = list(estimator.predict(input_fn=predict_input_fn))
    print(predictions)


if __name__ == '__main__':
    # regression()
    predict()

