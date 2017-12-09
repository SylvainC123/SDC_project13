from styx_msgs.msg import TrafficLight

import tensorflow as tf
import os
import cv2
import numpy as np
import rospy
import time

class TLClassifier(object):

	def load_graph(model_file):
  		graph = tf.Graph()
  		graph_def = tf.GraphDef()

  		with open(model_file, "rb") as f:
    		graph_def.ParseFromString(f.read())
  		with graph.as_default():
    		tf.import_graph_def(graph_def)
  		return graph
	
	def __init__(self):
				
		"""
        Creates tf session and loads calculation graph
        """
		# init default variables		
		
		tl_detector_dir =os.path.dirname(os.path.abspath(__file__))
		model_dir = os.path.join(tl_detector_dir,'models')
		self.model_file=os.path.join(model_dir,'retrained_graph.pb')

		label_path=os.path.join(model_dir,'image_labels.txt')
		self.label_file = label_path
	  	self.input_height = 224
	  	self.input_width = 224
	  	self.input_mean = 128
	  	self.input_std = 128
	  	self.input_layer = "input"
	  	self.output_layer = "final_result"

        # load graph
		self.graph = load_graph(self.model_file)

		# create session
		self.config = tf.ConfigProto()
		self.sess = tf.Session(graph=self.graph, config=self.config)	
		
		# inputs/outputs
	  	input_name = "import/" + self.input_layer
	  	output_name = "import/" + self.output_layer
	  	self.input_operation = graph.get_operation_by_name(input_name);
	  	self.output_operation = graph.get_operation_by_name(output_name);

		'''
		'''
		# load classifier
		self.model_path= model_path
		# load graph
		self.config = tf.ConfigProto()
		# **** may need to precise for GPU or CPU
		self.graph = _load_graph(self.model_path, self.config)
		# TF session
		self.sess = tf.Session(graph=self.graph, config=self.config)
		# 
		'''      
		pass
		'''
		

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
		
		# pre-process image
		float_caster = tf.cast(image, tf.float32)
  		dims_expander = tf.expand_dims(float_caster, 0);
  		resized = tf.image.resize_bilinear(dims_expander, [self.input_height, self.input_width])
  		normalized = tf.divide(tf.subtract(resized, [self.input_mean]), [self.input_std])
  		self.resized_t_img = self.sess.run(normalized)
				
		# run TF prediction		
		#start = time.time()
		results = sess.run(self.output_operation.outputs[0],
                      {self.input_operation.outputs[0]: self.resized_t_img})
		#end=time.time()
		results = np.squeeze(results)
		
		if result[0]>=result[1]:
			return TrafficLight.RED
		else:
			return TrafficLight.GREEN

		# note: first label is 'stop', second label is 'go'
        return TrafficLight.UNKNOWN
		
		#return TrafficLight.RED#GREEN
