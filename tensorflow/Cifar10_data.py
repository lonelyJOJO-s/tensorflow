import os
import tensorflow as tf

num_classes = 10  # 十分类问题
num_examples_for_train = 50000
num_examples_for_eval = 10000


# 用于返回读取的cifar10数据
class Cifar10Record(object):
    pass


#  只读入一张图片？  answer：实验表明：FixLengthRecordReader读取时会自动地匹配到未读取的字段
def read_Cifar10(file_queue):
    result = Cifar10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth  # 3072
    record_bytes = label_bytes + image_bytes  # 3073
    #  raeder读取时会随机读取，不一定按照顺序
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)
    #  decode_row 将字符串解析成图像对应的像素数组
    record_bytes = tf.decode_raw(value, tf.uint8)
    #  strided_slice()对input截取[begin, end)
    #  对record_bytes 截取[0 , label_bytes) 的数据 此处即截取第一个字节
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    #  note: record_bytes 以 depth, height, width 排序
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])
    #  转换格式
    result.unit8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def input(data_dir, batch_size, distorted):
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    #  train.string_input_producer 把文件输入到一个管道队列
    file_queue = tf.train.string_input_producer(filenames)
    read_input = read_Cifar10(file_queue)
    reshaped_image = tf.cast(read_input.unit8image, tf.float32)
    num_examples_per_epoch = num_examples_for_train
    #  对图像数据进行数据增强处理
    if distorted is not None:
        #  裁剪图片
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # flip the picture
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        #  random_brightness()调整亮度
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 调整对比度
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 标准化图片（非归一化)
        float_image = tf.image.per_image_standardization(adjusted_contrast)
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_example = int(num_examples_for_eval * 0.4)  # 4000
        print('Filling queue with %d CIFAR images before starting to train'
              'This will take a little time' % min_queue_example)
        # 使用shuffle_batch() 随机产生一个batch的 image & label  capacity 表示样本总容量， min_after_dequeue
        # 表示单次提取batch_size个样本后，剩余队列进行下一次提取保证的至少剩余量，如果太大，则刚开始需要向capacity中补充很多数据
        # 此处表示提取三次batch
        image_train, label_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                          num_threads=16, capacity=min_queue_example + 3 * batch_size,
                                                          min_after_dequeue=min_queue_example)
        return image_train, tf.reshape(label_train, [batch_size])
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_example = int(num_examples_for_eval * 0.4)  # 4000
        image_train, label_train = tf.train.batch([float_image, read_input.label], batch_size=batch_size,
                                                  num_threads=16, capacity=min_queue_example + 3 * batch_size)
        return image_train, tf.reshape(label_train, [batch_size])


