
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import os
from sklearn.utils import shuffle
tf.enable_eager_execution()

def load_sample(sample_dir, is_shuffle=True):
    print("loading sample dataset...")
    lfilenames = []
    labelsnames = []
    for dirpath, dirnames, filenames in os.walk(sample_dir):
        for filename in filenames:
            filename_path = dirpath + "\\" + filename
            lfilenames.append(filename_path)
            labelsnames.append(dirpath.split('\\')[-1])
    lab = list(sorted(set(labelsnames)))
    labdict = dict(zip(lab, list(range(len(lab)))))
    labels = [labdict[i] for i in labelsnames]
    if is_shuffle:
        return shuffle(np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)
    return (np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)


def _norm_image(image, size, ch=1, flatten_flag=False):
    image_decoded = image / 255.
    if flatten_flag:
        image_decoded = tf.reshape(image_decoded, shape=[size[0]*size[1]*ch])
    return image_decoded


def get_dataset(directory, size, batch_size):
    """

    :param directory: 图片路径
    :param size: 图片大小，手写字体是[28,28]
    :param batch_size:
    :return:
    """
    (filenames, labels), _ = load_sample(directory, is_shuffle=False)
    def _parseone(filename, label): # 解析一个图片文件
        image_string = tf.read_file(filename=filename)
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape([None, None, None])
        # [batch, new_height, new_width, channels]->[batch, new_height, new_width, channels]
        # [new_height, new_width, channels]->[new_height, new_width, channels] 这里是3D的（parseone作用于每一个图片）
        image_decoded = tf.image.resize(image_decoded, size)
        image_decoded = _norm_image(image_decoded, size)
        label = tf.cast(tf.reshape(label, []), dtype=tf.int32) # 将label转换为张量 tf.reshape(1, [])->tf.Tensor(1, shape=(), dtype=int32)
        return image_decoded, label
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(map_func=_parseone)
    dataset = dataset.repeat().batch(batch_size=batch_size)
    return dataset


class MNISTModel(tf.layers.Layer):
    def __init__(self, name):
        super(MNISTModel, self).__init__(name=name)

        self._input_shape = [-1, 28, 28, 1]
        # strides默认(1,1)
        self.conv1 = tf.layers.Conv2D(filters=32, kernel_size=5, activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu)
        # 注意tf.layers.Dense tf.layers.Conv2d和tf.layers.dense tf.layers.conv2d的区别
        # 一个是class和一个是function
        self.fc1 = tf.layers.Dense(units=1024, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(units=10) # 10分类
        self.dropout = tf.layers.Dropout(rate=0.5)
        # VALID不做0填充，有可能丢弃末尾数据，SAME，0填充，使得输出与输入大小相同
        self.max_pool2d = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")

    def call(self, inputs, training):
        x = tf.reshape(inputs, self._input_shape)
        x = self.conv1(x) # [-1, 28, 28, 32]
        x = self.max_pool2d(x)
        x = self.conv2(x) # [-1, 28, 28, 64]
        x = self.max_pool2d(x)
        x = tf.keras.layers.Flatten()(x) # [batch_size, -1]
        x = self.fc1(x)
        if training:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


def compute_loss(model, inputs, labels):
    predictions = model(inputs, training=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
    return tf.reduce_mean(loss)


def train():
    batch_size = 10
    num_epoches = 2
    data_size = 500*10
    display_iter = 100
    dir = r"E:\code\python\deeplearning\tensorflow1.x\data\mnist_digits_images"
    save_dir = r"E:\\code\\python\\deeplearning\\tensorflow1.x\\data\\ck\\"
    dataset = get_dataset(directory=dir, size=[28, 28], batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()
    model = MNISTModel(name='net')
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    grad_fn = tfe.implicit_gradients(compute_loss)

    while global_step * batch_size / data_size < num_epoches:
        step = int(global_step * batch_size / data_size)
        x, y = tf.cast(data[0], dtype=tf.float32), data[1]
        grads_and_vars = grad_fn(model, x, y)
        optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=tf.train.get_or_create_global_step())
        # 获取要保存的变量
        if global_step % display_iter == 0:
            all_variables = (model.variables + optimizer.variables() + [global_step])
            tfe.Saver(all_variables).save(save_dir, global_step=global_step) # 检查点文件
        print("Epoch:{}, Iteration:{}, loss:{}".format(step, global_step, compute_loss(model, x, y)))
        global_step = tf.train.get_or_create_global_step()


if __name__ == '__main__':
    train()