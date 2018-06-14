# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# 将分类名称转成ID号
def class_text_to_int(row_label):
    if row_label == 'AAaa0007014':
        return 1
    elif row_label == 'AAaa0007024':
        return 2
    elif row_label == 'AAaa0007008':
        return 3
    elif row_label == 'AAaa0007015':
        return 4
    elif row_label == 'AAaa0007005':
        return 5
    elif row_label == 'AAaa0007002':
        return 6
    elif row_label == 'AAaa0001002':
        return 7
    elif row_label == 'AAaa0010001':
        return 8
    elif row_label == 'AIxx0001001':
        return 9
    elif row_label == 'ALab0014002':
        return 10
    elif row_label == 'ALac0043003':
        return 11
    elif row_label == 'ALac0021001':
        return 12
    elif row_label == 'ALac0022001':
        return 13
    elif row_label == 'ALab0019003':
        return 14
    elif row_label == 'ALac0023004':
        return 15
    elif row_label == 'ADaa0003004':
        return 16
    elif row_label == 'ADaa0001003':
        return 17
    elif row_label == 'AKae0013001':
        return 18
    elif row_label == 'AKae0012001':
        return 19
    elif row_label == 'AKae0027003':
        return 20
    elif row_label == 'AKae0035001':
        return 21
    elif row_label == 'AKae0037001':
        return 22
    elif row_label == 'AKae0021005':
        return 23
    elif row_label == 'AKae0021003':
        return 24
    elif row_label == 'AKad0002001':
        return 25
    elif row_label == 'AKac0018001':
        return 26
    elif row_label == 'AKae0040002':
        return 27
    elif row_label == 'AKac0011001':
        return 28
    elif row_label == 'AKae0019001':
        return 29
    elif row_label == 'AKae0038002':
        return 30
    elif row_label == 'AFae0005001':
        return 31
    elif row_label == 'AFae0022001':
        return 32
    elif row_label == 'AFab0014005':
        return 33
    elif row_label == 'AFae0021001':
        return 34
    elif row_label == 'AFae0001002':
        return 35
    elif row_label == 'AFae0020003':
        return 36
    elif row_label == 'AFae0021006':
        return 37
    elif row_label == 'AFab0001037':
        return 38
    elif row_label == 'ACab0009002':
        return 39
    elif row_label == 'ACab0011001':
        return 40
    elif row_label == 'ACab0008009':
        return 41
    elif row_label == 'ACab0005009':
        return 42
    elif row_label == 'ACaa0004002':
        return 43
    elif row_label == 'ACaa0003007':
        return 44
    elif row_label == 'ACaa0003002':
        return 45
    elif row_label == 'ACac0001001':
        return 46
    elif row_label == 'ACab0009003':
        return 47
    elif row_label == 'ACab0005005':
        return 48
    elif row_label == 'ACab0015003':
        return 49
    elif row_label == 'ACab0005001':
        return 50
    elif row_label == 'ACab0008002':
        return 51
    elif row_label == 'ACaa0001002':
        return 52
    elif row_label == 'ACaa0003004':
        return 53
    elif row_label == 'AMxx0001002':
        return 54
    elif row_label == 'AMxx0001003':
        return 55
    elif row_label == 'AMxx0001001':
        return 56
    elif row_label == 'AJaa0003006':
        return 57
    elif row_label == 'AGaf0019004':
        return 58
    elif row_label == 'AGae0011001':
        return 59
    elif row_label == 'AGae0012003':
        return 60
    elif row_label == 'AGai0006002':
        return 61
    elif row_label == 'AGaf0005009':
        return 62
    elif row_label == 'AGae0008001':
        return 63
    elif row_label == 'AGae0021001':
        return 64
    elif row_label == 'AGaf0020002':
        return 65
    elif row_label == 'AGac0001002':
        return 66
    elif row_label == 'AGae0017003':
        return 67
    elif row_label == 'AGaf0006013':
        return 68
    elif row_label == 'AGae0018001':
        return 69
    elif row_label == 'AGai0007001':
        return 70
    elif row_label == 'AGai0009001':
        return 71
    elif row_label == 'AGad0001001':
        return 72
    elif row_label == 'AGaj0001002':
        return 73
    elif row_label == 'AGae0007001':
        return 74
    elif row_label == 'AGai0016010':
        return 75
    elif row_label == 'AGai0011001':
        return 76
    elif row_label == 'AGai0011002':
        return 77
    elif row_label == 'AGae0009001':
        return 78
    elif row_label == 'AGai0005001':
        return 79
    elif row_label == 'AGaf0006002':
        return 80
    elif row_label == 'AGae0017002':
        return 81
    elif row_label == 'AGai0011006':
        return 82
    elif row_label == 'AGae0016001':
        return 83
    elif row_label == 'AGae0015001':
        return 84
    elif row_label == 'AGaf0019047':
        return 85
    elif row_label == 'AGaf0015001':
        return 86
    elif row_label == 'AGaf0019038':
        return 87
    elif row_label == 'AGai0009002':
        return 88
    elif row_label == 'AGai0016009':
        return 89
    elif row_label == 'AGaf0003010':
        return 90
    elif row_label == 'ALac0033001':
        return 91
    elif row_label == 'AFab0019003':
        return 92
    elif row_label == 'ADaa0006001':
        return 93
    elif row_label == 'ALac0003001':
        return 94
    else:
        print('NONE: ' + row_label)
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    print(os.path.join(path, '{}'.format(group.filename)))
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = (group.filename).encode('utf_8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf_8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(csv_input, output_path, imgPath):
    writer = tf.python_io.TFRecordWriter(output_path)
    path = imgPath
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':

    imgPath = '...ImagesPATH'

    # 生成train.record文件
    output_path = '...data_train.record'
    csv_train_input = '...\\train.csv'
    main(csv_train_input, output_path, imgPath)

    # 生成验证文件 eval.record
    output_path = '...data_eval.record'
    csv_eval_input = '...\\eval.csv'
    main(csv_eval_input, output_path, imgPath)
