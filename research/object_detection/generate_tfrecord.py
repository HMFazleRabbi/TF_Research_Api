"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=images/train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=images/test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import shutil
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label.upper() == 'BODY':
        return 1
    elif row_label.upper() == 'PIN':
        return 2
    else:
        raise IOError
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # Debug
    dir_name="./debug/TF_Record_Images_Labelled-20200414_1147-wiegst"
    dir_ori_name="./debug/TF_Record_Images_ORI-20200414_1147-wiegst"
    if not (os.path.isdir(dir_name)):
        os.makedirs(dir_name)
    if not (os.path.isdir(dir_ori_name)):
        os.makedirs(dir_ori_name)
    img_path = os.path.join (dir_name, group.filename)
    img_buffer =cv2.imread(os.path.join(path, '{}'.format(group.filename)))

    for index, row in group.object.iterrows():
        # Sanity check
        if (row['xmax']<row['xmin']) or (row['ymax'] <= row['ymin']):
            raise IOError
        
        if not (str(row['xmax']).isnumeric() and str(row['xmin']).isnumeric() and str(row['ymax']).isnumeric() and str(row['ymin'])):
            raise IOError

        # Store
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        # classes_text.append(row['class'].encode('utf8'))
        classes_text.append(row['class'].upper().encode('utf8'))
        classes.append(class_text_to_int(row['class']))

        # Debug
        p1=(int (row['xmin']),int (row['ymin']))
        p2=(int (row['xmax']),int (row['ymax']))
        cv2.rectangle(img_buffer, p1,p2, (0,255,0), 1)

    if len(classes) < 1:
        raise IOError

    # Selective option to skip
    elif len(classes) < -1:
        return None

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

    # Debug
    shutil.copy(os.path.join(path, '{}'.format(group.filename)), dir_ori_name)
    cv2.imwrite(img_path, img_buffer)

    # Return
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    # examples = pd.read_csv(FLAGS.csv_input).sample(frac=1).reset_index(drop=True)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        if (tf_example ==None):
            continue
        else:
            writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
