
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def load_sample(sample_dir):
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
    return shuffle(np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)

def get_batches(image, label, resize_w, resize_h, channels, batch_size):
    queue = tf.train.slice_input_producer([image, label]) # 输入队列
    image_c = tf.read_file(queue[0]) # 从输入队列里读取image路径
    image = tf.image.decode_bmp(image_c, channels)
    label = queue[1]
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h) # 修改图片大小
    image = tf.image.per_image_standardization(image) # 标准化
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64) # 生成批次数据
    image_batches = tf.cast(image_batch, dtype=tf.float32)
    label_batches = tf.reshape(label_batch, [batch_size])
    return image_batches, label_batches

def show_result(subplot, title, thisimg):
    p = plt.subplot(subplot) #子图
    p.axis('off')
    p.imshow(np.reshape(thisimg, newshape=(28, 28)))
    p.set_title(title)

def show_img(index, img, label, ntop):
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    print(index)
    ntop = min(ntop, 9)
    for i in range(ntop):
        show_result(100+10*ntop+i+1, label[i], img[i]) # 一行 nstop列，第1+i个
    plt.show()

def main1():
    (image, label), labelnames = load_sample(r"E:\code\python\deeplearning\tensorflow1.x\data\mnist_digits_images")
    batch_size = 16
    image_batches, label_batches = get_batches(image, label, 28, 28, 1, batch_size=batch_size)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator() # 建立队列协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 启动队列线程
        try:
            for step in range(10):
                if coord.should_stop():
                    break
                images, labels = sess.run([image_batches, label_batches]) # 注入数据
                show_img(step, images, labels, batch_size)
                print(labels)
        except tf.errors.OutOfRangeError:
            print("Done!!!")
        finally:
            coord.request_stop()
        coord.join((threads))

if __name__ == '__main__':
    main1()