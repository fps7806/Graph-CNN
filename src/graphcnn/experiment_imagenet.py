from graphcnn.helper import *
from graphcnn.network import *
from graphcnn.layers import *
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
import glob
import time
from graphcnn.experiment import GraphCNNExperiment, make_batch_queue
from tensorflow.python.training import queue_runner

# This class is responsible for setting up and running experiments
# Also provides helper functions related to experiments (e.g. get accuracy)
class GraphCNNImageNetExperiment(GraphCNNExperiment):
    def __init__(self, dataset_name, model_name, net_constructor):
        GraphCNNExperiment.__init__(self, dataset_name, model_name, net_constructor)
        
        self.number_of_classes = 1000
        
        self.image_resize_width = 256
        self.image_resize_height = 256
        
        self.image_width = 227
        self.image_height = 227
    # Create input_producers and batch queues
    def create_data(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                # Create the training queue
                with tf.variable_scope('train_data') as scope:
                    self.print_ext('Creating training Tensorflow Tensors')
                    
                    filenames = []
                    labels = []
                    with open(self.train_list_file) as file:
                        for line in file:
                            key, value = line[:-1].split()
                            value = int(value)
                            if value < self.number_of_classes:
                                labels.append(value)
                                filenames.append(key)
                    training_samples = [np.array(filenames), np.array(labels).astype(np.int64)]
                    training_samples = self.create_input_variable(training_samples)
                    single_sample = tf.train.slice_input_producer(training_samples, shuffle=True, capacity=2048)
                    
                    single_sample[0] = tf.image.decode_jpeg(tf.read_file(single_sample[0]), channels=3)
                    single_sample[0] = tf.random_crop(tf.image.resize_images(single_sample[0], [self.image_resize_width, self.image_resize_height]), [self.image_width, self.image_height, 3])
                    single_sample[0] = tf.image.random_flip_left_right(single_sample[0])
                    single_sample[0] = tf.cast(single_sample[0], dtype=tf.float32)/255
                    train_queue = make_batch_queue(single_sample, capacity=self.train_batch_size*2, num_threads=8)

                # Create the test queue
                with tf.variable_scope('test_data') as scope:
                    self.print_ext('Creating test Tensorflow Tensors')
                    
                    filenames = []
                    labels = []
                    with open(self.val_list_file) as file:
                        for line in file:
                            key, value = line[:-1].split()
                            value = int(value)
                            if value < self.number_of_classes:
                                labels.append(value)
                                filenames.append(key)
                    test_samples = [np.array(filenames), np.array(labels).astype(np.int64)]
                    test_samples = self.create_input_variable(test_samples)
                    single_sample = tf.train.slice_input_producer(test_samples, shuffle=True, capacity=128)
                    
                    single_sample[0] = tf.image.decode_jpeg(tf.read_file(single_sample[0]), channels=3)
                    single_sample[0] = tf.image.resize_image_with_crop_or_pad(tf.image.resize_images(single_sample[0], [self.image_resize_width, self.image_resize_height]), self.image_width, self.image_height)
                    single_sample[0].set_shape([self.image_width, self.image_height, 3])
                    single_sample[0] = tf.cast(single_sample[0], dtype=tf.float32)/255
                    test_queue = make_batch_queue(single_sample, capacity=self.test_batch_size*2, num_threads=1)
                        
                result = tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_queue.dequeue_many(self.test_batch_size))

                # Have to add placeholder for A and mask
                result = [result[0], None, result[1], None]
                return result