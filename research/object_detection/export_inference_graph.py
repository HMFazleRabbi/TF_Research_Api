# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Tool to export an object detection model for inference.

Prepares an object detection tensorflow graph for inference using model
configuration and a trained checkpoint. Outputs inference
graph, associated checkpoint files, a frozen inference graph and a
SavedModel (https://tensorflow.github.io/serving/serving_basic.html).

The inference graph contains one of three input nodes depending on the user
specified option.
  * `image_tensor`: Accepts a uint8 4-D tensor of shape [None, None, None, 3]
  * `encoded_image_string_tensor`: Accepts a 1-D string tensor of shape [None]
    containing encoded PNG or JPEG images. Image resolutions are expected to be
    the same if more than 1 image is provided.
  * `tf_example`: Accepts a 1-D string tensor of shape [None] containing
    serialized TFExample protos. Image resolutions are expected to be the same
    if more than 1 image is provided.

and the following output nodes returned by the model.postprocess(..):
  * `num_detections`: Outputs float32 tensors of the form [batch]
      that specifies the number of valid boxes per image in the batch.
  * `detection_boxes`: Outputs float32 tensors of the form
      [batch, num_boxes, 4] containing detected boxes.
  * `detection_scores`: Outputs float32 tensors of the form
      [batch, num_boxes] containing class scores for the detections.
  * `detection_classes`: Outputs float32 tensors of the form
      [batch, num_boxes] containing classes for the detections.
  * `raw_detection_boxes`: Outputs float32 tensors of the form
      [batch, raw_num_boxes, 4] containing detection boxes without
      post-processing.
  * `raw_detection_scores`: Outputs float32 tensors of the form
      [batch, raw_num_boxes, num_classes_with_background] containing class score
      logits for raw detection boxes.
  * `detection_masks`: (Optional) Outputs float32 tensors of the form
      [batch, num_boxes, mask_height, mask_width] containing predicted instance
      masks for each box if its present in the dictionary of postprocessed
      tensors returned by the model.
  * detection_multiclass_scores: (Optional) Outputs float32 tensor of shape
      [batch, num_boxes, num_classes_with_background] for containing class
      score distribution for detected boxes including background if any.
  * detection_features: (Optional) float32 tensor of shape
      [batch, num_boxes, roi_height, roi_width, depth]
  containing classifier features

Notes:
 * This tool uses `use_moving_averages` from eval_config to decide which
   weights to freeze.

Example Usage:
--------------
python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory

Export Command History
----------------------
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/Checkpoint-sagen_dich/model.ckpt-162857 --output_directory inference_graph/V-10-sagen_dich-160000
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/Checkpoint-wiegst\model.ckpt-50000 --output_directory inference_graph/V-10-wiegst --config_override "model{faster_rcnn {second_stage_post_processing {batch_non_max_suppression {score_threshold: 0.3 }}}}"
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/Checkpoint-wichtig\model.ckpt-10000 --output_directory inference_graph/V-11-wichtig --config_override "model{faster_rcnn {second_stage_post_processing {batch_non_max_suppression {score_threshold: 0.3 }}}}"
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/Checkpoint-frittieren\model.ckpt-30000 --output_directory inference_graph/V-12-frittieren --config_override "model{faster_rcnn {second_stage_post_processing {batch_non_max_suppression {score_threshold: 0.3 }}}}"
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-gekochtes-5k\model.ckpt-5000 --output_directory inference_graph/V-13-gekochtes-5k
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-gekochtes-25k\model.ckpt-25000 --output_directory inference_graph/V-13-gekochtes-25k
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-bezahlen-1/model.ckpt-20000 --output_directory inference_graph/V-May-01-Checkpoint-bezahlen-1
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-bezahlen-3/model.ckpt-16539 --output_directory inference_graph/V-May-02-Checkpoint-bezahlen-3
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Ausgabe-1/model.ckpt-50000 --output_directory inference_graph/V-May-03-Checkpoint-Ausgabe-1
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Ihr-2/model.ckpt-75000 --output_directory inference_graph/V-May-04-Checkpoint-Ihr-2
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Ihr-3/model.ckpt-75000 --output_directory inference_graph/V-May-04-Checkpoint-Ihr-3
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Ihr-4/model.ckpt-75000 --output_directory inference_graph/V-May-05-Checkpoint-Ihr-4
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Ihr-7/model.ckpt-200000 --output_directory inference_graph/V-Jun-04-Checkpoint-Ihr-7
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Geschlafen-1/model.ckpt-29097 --output_directory inference_graph/V-Jun-16-Checkpoint-Geschlafen-1
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Geschlafen-1/model.ckpt-200000 --output_directory inference_graph/V-Jun-16-Checkpoint-Geschlafen-1-200000
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/local-faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/Checkpoint-Mantell-7/model.ckpt-600000 --output_directory inference_graph/V-Jun-25-Checkpoint-Mantell-7-600K
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/Checkpoint-Mantell-9/Mantell-9.config --trained_checkpoint_prefix training/Checkpoint-Mantell-9/model.ckpt-600000 --output_directory inference_graph/V-Jun-25-Checkpoint-Mantell-9-600K
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/Checkpoint-Mantell-13-300k/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/Checkpoint-Mantell-13-300k/model.ckpt-304415 --output_directory inference_graph/V-Jun-30-Checkpoint-Mantell-13-300k
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/Checkpoint-Mantell-13/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/Checkpoint-Mantell-13/model.ckpt-900000 --output_directory inference_graph/V-Jul-02-Checkpoint-Mantell-13
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/Checkpoint-Hemd-5/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Hemd-5/model.ckpt-72090 --output_directory inference_graph/V-Jul-11-Checkpoint-Hemd-5
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/Checkpoint-Hemd-6/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Hemd-6/model.ckpt-75000 --output_directory inference_graph/V-Jul-12-Checkpoint-Hemd-6
python export_inference_graph.py --input_type image_tensor  --pipeline_config_path training/Checkpoint-Hemd-8/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/Checkpoint-Hemd-8/model.ckpt-60527 --output_directory inference_graph/V-Jul-14-Checkpoint-Hemd-8



    

The expected output would be in the directory
path/to/exported_model_directory (which is created if it does not exist)
with contents:
 - inference_graph.pbtxt
 - model.ckpt.data-00000-of-00001
 - model.ckpt.info
 - model.ckpt.meta
 - frozen_inference_graph.pb
 + saved_model (a directory)

Config overrides (see the `config_override` flag) are text protobufs
(also of type pipeline_pb2.TrainEvalPipelineConfig) which are used to override
certain fields in the provided pipeline_config_path.  These are useful for
making small changes to the inference graph that differ from the training or
eval config.

Example Usage (in which we change the second stage post-processing score
threshold to be 0.5):

python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory \
    --config_override " \
            model{ \
              faster_rcnn { \
                second_stage_post_processing { \
                  batch_non_max_suppression { \
                    score_threshold: 0.5 \
                  } \
                } \
              } \
            }"
"""
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')
flags.DEFINE_string('input_shape', None,
                    'If input_type is `image_tensor`, this can explicitly set '
                    'the shape of this input tensor to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. If not specified, for an `image_tensor, the '
                    'default shape will be partially specified as '
                    '`[None, None, None, 3]`.')
flags.DEFINE_string('pipeline_config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')
flags.DEFINE_string('config_override', '',
                    'pipeline_pb2.TrainEvalPipelineConfig '
                    'text proto to override pipeline_config_path.')
flags.DEFINE_boolean('write_inference_graph', False,
                     'If true, writes inference graph to disk.')
tf.app.flags.mark_flag_as_required('pipeline_config_path')
tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
tf.app.flags.mark_flag_as_required('output_directory')
FLAGS = flags.FLAGS


def main(_):
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  text_format.Merge(FLAGS.config_override, pipeline_config)
  if FLAGS.input_shape:
    input_shape = [
        int(dim) if dim != '-1' else None
        for dim in FLAGS.input_shape.split(',')
    ]
  else:
    input_shape = None
  exporter.export_inference_graph(
      FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_prefix,
      FLAGS.output_directory, input_shape=input_shape,
      write_inference_graph=FLAGS.write_inference_graph)


if __name__ == '__main__':
  tf.app.run()
