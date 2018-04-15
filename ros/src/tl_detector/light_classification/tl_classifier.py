import os

import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight

# Traffic Light Detection Based on Tensorflow demo shown here:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

SELF_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(SELF_PATH, "..", "..", "..", "classifier", "model.pb")


class TLClassifier(object):
    def __init__(self):
        # Load tensorflow graph into memory
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.num = self.graph.get_tensor_by_name('num_detections:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.session = tf.Session(graph=self.graph, config=config)

    def run_inference_for_single_image(self, image):
        with self.graph.as_default():
            # Run inference
            number, boxes, scores, classes = self.session.run(
                [self.num, self.boxes, self.scores, self.classes],
                feed_dict={self.image_tensor: np.expand_dims(image, 0)}
            )
        return number[0], boxes[0], scores[0], classes[0].astype(int)

    @staticmethod
    def filter_output_for_class(number, boxes, scores, classes, cls=10):
        mask = classes == cls
        return number, boxes[mask], scores[mask], classes[mask]

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        number, boxes, scores, classes = self.run_inference_for_single_image(image)
        predict, threshold = classes[0], scores[0]

        if threshold < 0.3:
            return TrafficLight.UNKNOWN

        if predict == 1:
            return TrafficLight.RED
        elif predict == 2:
            return TrafficLight.YELLOW
        elif predict == 3:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN
