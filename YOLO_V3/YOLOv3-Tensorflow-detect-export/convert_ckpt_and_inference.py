# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import time
import uff
import tensorrt as trt

from yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', 'images/giraffe.jpg', 'Input image')
tf.app.flags.DEFINE_string('output_img', 'out/detected.jpg', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'model_data/coco_classes.txt', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'weight/yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def draw_boxes(boxes, img, cls_names, size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)


def convert_to_original_size(box, size, original_size):
    ratio = 1.0*original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def main(argv=None):
    
    BASE_PATH = 'images'
    TEST_IMAGES = os.listdir(BASE_PATH)
    TEST_IMAGES.sort()
    print(TEST_IMAGES)
    
    
#     img = Image.open(FLAGS.input_img)
#     w,h = img.size
#     img_resized = img.resize(size=(FLAGS.size, FLAGS.size))
    with tf.Graph().as_default():
        classes = load_coco_names(FLAGS.class_names)

        # placeholder for detector inputs
        inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

        with tf.variable_scope('detector'):
            detections = yolo_v3(inputs, len(classes), data_format='NHWC')#Tensor("detector/yolo-v3/concat:0", shape=(?, 10647, 85), dtype=float32)
            load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

        boxes = detections_boxes(detections)#shape=(?, 10647, 85), dtype=float32)
        #coordinates of top left and bottom right points+num_class_confidence

        boxes = tf.identity(boxes, name='boxes')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(load_ops)

            writer =tf.summary.FileWriter("logs/",graph = sess.graph)
            writer.close()
            saver.save(sess,"models/yolov3.ckpt")
            
            for img in TEST_IMAGES:
                start = time.time()
                image_path = os.path.join(BASE_PATH, img)

                image = Image.open(image_path)
                w,h = image.size
                img_resized = image.resize(size=(FLAGS.size, FLAGS.size))
              
                detected_boxes = sess.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})

                filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold,
                                             iou_threshold=FLAGS.iou_threshold)

                draw_boxes(filtered_boxes, image, classes, (FLAGS.size, FLAGS.size))
                
                #plt.imshow(image)
                #plt.show()

                image.save(FLAGS.output_img)
                print time.time()-start
            #print boxes.name, detections.name
            OUTPUT_NAMES = ['boxes']
            graphdef = tf.get_default_graph().as_graph_def()
            frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                            graphdef,
                                                                            OUTPUT_NAMES)
            return tf.graph_util.remove_training_nodes(frozen_graph)
            print OUTPUT_NAMES

if __name__ == '__main__':

    # refer to this
    # https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/topics/topics/workflows/tf_to_tensorrt.html#Convert-a-Tensorflow-Model-to-UFF

    tf_model = tf.app.run()
    uff_model = uff.from_tensorflow(tf_model, ["boxes"])
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

    parser = uffparser.create_uff_parser()
    parser.register_input("Placeholder", (1,28,28))
    parser.register_output("boxes")
    engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 20)

    host_mem = parser.hidden_plugin_memory()


    parser.destroy()


    # trt_graph = trt.create_inference_graph(
    # input_graph_def = tf_model,
    # outputs = ['boxes'],
    # max_batch_size=16,
    # max_workspace_size_bytes=3000000000,
    # precision_mode=FP16,
    # minimum_segment_size=3)