
import tensorflow as tf
import numpy as np

def generate_data(datasize=100):
    train_X = np.linspace(-1, 1, datasize)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
    return train_X, train_Y

def generate_dataset(X, Y):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y)) # 元组形式
    dataset2 = tf.data.Dataset.from_tensor_slices({
        "x": X,
        "y": Y
    })
    dataset3 = dataset.repeat().batch(batch_size=10) # 可重复无数次，每一个取出一个批次（大小为10）
    dataset4 = dataset2.map(lambda data:(data['x'], tf.cast(data['y'], dtype=tf.int32)))
    dataset5 = dataset.shuffle(100)
    return dataset3

# 注意，tf.data.Dataset.from_tensor_slices(inputs)，如果输入是list，
# 例如[1,2,3,4,5]，则系统将其看做5个样本数据array([1,2,3,4,5])
# 如果inputs是元组，则系统将其看做一个样本（包含5列）

def get_one(dataset):
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next() # 取出一条数据或者一个batch
    return data

def main1():
    X, y = generate_data()
    dataset = generate_dataset(X, y)
    data = get_one(dataset)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for _ in range(10):
            data_ = sess.run(data)
            print(data_)

if __name__ == '__main__':
    main1()