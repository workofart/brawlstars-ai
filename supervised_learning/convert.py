import numpy as np
import tensorflow as tf
import cv2

RAW_SCREEN_PATH = 'data/training_data_bounty_attack_raw_screen_200_200.npy'
FEATURE_PATH = 'data/training_data_bounty_attack_mnet2.npy'
IMAGE_INPUT_TENSOR = 'FeatureExtractor/MobilenetV2/MobilenetV2/input:0'
# 'image_tensor:0'
FEATURE_EXTRACTOR_TENSOR = 'FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_5_3x3_s2_128/Relu6:0'
# FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6:0

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'D:\\Github\\brawlstars-ai\\object_detection\\all_inference_graph\\frozen_inference_graph.pb'

def screen2feature(screen, sess, detection_graph):
    # Definite input and output Tensors for detection_graph
    # Input shape: (12x300x300x3)
    image_tensor = detection_graph.get_tensor_by_name(IMAGE_INPUT_TENSOR)
    feature_vector = detection_graph.get_tensor_by_name(
        FEATURE_EXTRACTOR_TENSOR)
    converted_input = cv2.resize(screen, (200,200))
    # Feature Extraction (Output Shape: 12x1x1x128)
    (rep) = sess.run(
        [feature_vector],
        feed_dict={image_tensor: np.expand_dims(converted_input, axis=0)})
    return rep

def create_sess(config):
    # Create the graph definition for the mobilenet model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        return tf.Session(graph=detection_graph, config=config), detection_graph
    

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True

    training_data = np.load(RAW_SCREEN_PATH)
    print('loaded Numpy raw screen')

    sess, detection_graph = create_sess(config)
    print('Imported graph definition')

    output_data = []
    for idx in range(len(training_data)):
        screen = training_data[idx][0]
        movement = training_data[idx][1]
        action = training_data[idx][2]
        screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2RGB)
        feature = screen2feature(screen, sess, detection_graph)
        output_data.append([feature, movement, action])

    np.save(FEATURE_PATH, output_data)
    
# main()