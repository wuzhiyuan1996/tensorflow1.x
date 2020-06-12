
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
import numpy as np
from action.chapter04.testTFDS import load_sample
from PIL import Image


def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0.0]]*10
    csvColumn = tf.decode_csv(value, defaults)
    featureColumn = [i for i in csvColumn[1:-1]]
    labelColumn = csvColumn[-1]
    return tf.stack(featureColumn), labelColumn


def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    feature, label = read_data(file_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    feature_batch, label_batch = tf.train.shuffle_batch( # 随机取一个batch
        [feature, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return feature_batch, label_batch


def make_tfrecord(filenames, labels):
    writer = tf.python_io.TFRecordWriter(r"E:\code\python\deeplearning\tensorflow1.x\data\tfrecords\mydata.tfrecords")
    for i in tqdm(range(0, len(labels))):
        img = Image.open(filenames[i])
        img = img.resize((256, 256))
        img_raw = img.tobytes() # 将图片转换为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))
        }))
        writer.write(example.SerializeToString()) # 序列化为字符串
    writer.close()


def read_and_decode(filenames, flag='train', batch_size=3):
    if flag == 'train':
        filename_queue = tf.train.string_input_producer(filenames)
    else:
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False) # 测试数据当做一个epoch
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_examples,
                            features={
                                'image_raw':tf.FixedLenFeature([], tf.string),
                                'label':tf.FixedLenFeature([], tf.int64)
                            })
    images = tf.decode_raw(features['image_raw'], out_type=tf.uint8) # 将字符串解析成图像对应的像素数组
    images = tf.reshape(images, [256, 256, 3])
    labels = tf.cast(features['label'], tf.int32)
    if flag == 'train':
        images = tf.cast(images, tf.float32) * (1.0 / 255) - 0.5 # 归一化
        img_batch, label_batch = tf.train.batch([images, labels], batch_size=batch_size, capacity=20)
        return img_batch, label_batch
    return images, labels

def main1():
    # (filenames, labels), labelnames = load_sample(r"E:\code\python\deeplearning\tensorflow1.x\data\man_woman")
    # make_tfrecord(filenames=filenames, labels=labels)
    images, labels = read_and_decode([r"E:\code\python\deeplearning\tensorflow1.x\data\tfrecords\mydata.tfrecords"], flag='test')
    save_img_path = r"E:\\code\\python\\deeplearning\\tensorflow1.x\\data\\output\\"
    if tf.gfile.Exists(save_img_path): # 存在路径则删除
        tf.gfile.DeleteRecursively(save_img_path)
    tf.gfile.MakeDirs(save_img_path)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        myset = set([])
        try:
            i=0
            while True:
                images_, labels_ = sess.run([images, labels])
                labels_ = str(labels_)
                if labels_ not in myset:
                    myset.add(labels_)
                    tf.gfile.MakeDirs(save_img_path+labels_)
                img = Image.fromarray(images_, 'RGB')
                img.save(save_img_path+labels_+'\\'+str(i)+'_Label_.jpg')
                print(i)
                i = i + 1
        except tf.errors.OutOfRangeError:
            print("Done!!!")
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main1()