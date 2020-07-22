"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=images/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_05/train_labels.csv --image_dir=images/H_Dataset_05/train --output_path=images/H_Dataset_05/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_05/gray/train_labels.csv --image_dir=images/H_Dataset_05/gray/train --output_path=images/H_Dataset_05/gray/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_05/pintype/train_labels.csv --image_dir=images/H_Dataset_05/pintype/train --output_path=images/H_Dataset_05/pintype/train.record
  python generate_tfrecord.py --csv_input=D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/images/H_Dataset_03/Mixed/merged_train_labels_lt_800.csv --image_dir=D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/images/H_Dataset_03/Mixed/train_labels_lt_800 --output_path=images/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_05/pintype/train_labels.csv --image_dir=images/H_Dataset_05/pintype/train --output_path=images/H_Dataset_05/pintype/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_06/train_labels.csv --image_dir=images/H_Dataset_06/train --output_path=images/H_Dataset_06/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_07/extracted/MERGED/train_labels.csv --image_dir=images/H_Dataset_07/extracted/MERGED/train --output_path=images/H_Dataset_07/extracted/MERGED/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_08/H_Dataset_02_PinArray/QFP+SOIC/train_labels.csv --image_dir=images/H_Dataset_08/H_Dataset_02_PinArray/QFP+SOIC/train --output_path=images/H_Dataset_08/H_Dataset_02_PinArray/QFP+SOIC/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC/train_labels.csv --image_dir=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC/train --output_path=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC-Fragmented/train_labels.csv --image_dir=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC-Fragmented/train --output_path=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC-Fragmented/train.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_08/H_Dataset_02_PinOnly/Testcase-lt-600/train_labels.csv --image_dir=images/H_Dataset_08/H_Dataset_02_PinOnly/Testcase-lt-600/train --output_path=images/H_Dataset_08/H_Dataset_02_PinOnly/Testcase-lt-600/train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=images/test.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_05/test_labels.csv --image_dir=images/H_Dataset_05/test --output_path=images/H_Dataset_05/test.record

  python generate_tfrecord.py --csv_input=D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/images/H_Dataset_03/Mixed/merged_test_labels_lt_800.csv  --image_dir=D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/images/H_Dataset_03/Mixed/test_labels_lt_800 --output_path=images/test.record
  python generate_tfrecord.py --csv_input=D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/images/H_Dataset_03/Mixed/merged_train+test_labels_lt_800.csv  --image_dir=D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/images/H_Dataset_03/Mixed/merged_train+test_labels_lt_800 --output_path=images/merged_train+test_labels_lt_800.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_05/pintype/test_labels.csv --image_dir=images/H_Dataset_05/pintype/test --output_path=images/H_Dataset_05/pintype/test.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_06/test_labels.csv --image_dir=images/H_Dataset_06/test --output_path=images/H_Dataset_06/test.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC/test_labels.csv --image_dir=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC/test --output_path=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC/test.record
  python generate_tfrecord.py --csv_input=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC-Fragmented/test_labels.csv --image_dir=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC-Fragmented/test --output_path=images/H_Dataset_08/H_Dataset_02_PinOnly/QFP+SOIC-Fragmented/test.record


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
from utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

#############################################
#   Global Varialble
#############################################
# Default
# label_dictionary ={
#     'PIN': 0,
#     'BODY': 1
# }
# label_dictionary ={
#     'PIN': 2,
# }
# label_dictionary ={
#     'PIN': 0,
#     'BODY': 1,
#     'PIN_NL': 2,
#     'PIN_FLAT': 3,
#     'PIN_GULL': 4,
#     'PIN_JLEAD':5
    
# }


# Path to label map file
PATH_TO_LABELS = 'D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/data/pintype_detection_label_map.pbtxt'
PATH_TO_LABELS = 'D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/data/pinarray_detection_label_map.pbtxt'
PATH_TO_LABELS = 'D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/data/pinonly_detection_label_map.pbtxt'
PATH_TO_LABELS = 'D:/FZ_WS/JyNB/TF_Research_Api_LD_2_0/research/object_detection/data/lead_detection_label_map.pbtxt'

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_dictionary=label_map_util.get_label_map_dict(PATH_TO_LABELS)


# TO-DO replace this with label map
def class_text_to_int(row_label):
    
    if row_label.upper() in label_dictionary:
        return label_dictionary[row_label.upper()]
    else:
        raise IOError
        None

    # Commented by fazle@20200522_1115 and switching to label dictionary
    # if row_label.upper() == 'BODY':
    #     return 1
    # elif row_label.upper() == 'PIN':
    #     return 2
    # else:
    #     raise IOError
    #     None

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
    dir_name="./debug/TF_Record_Images_Labelled-20200520_1806"
    dir_ori_name="./debug/TF_Record_Images_ORI-20200520_1806"
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
