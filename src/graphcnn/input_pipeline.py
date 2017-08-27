import tensorflow as tf
import numpy as np
from .helper import *
from sklearn.model_selection import KFold
from .flags import FLAGS

class InputPipeline(object):

    # Prepares samples for experiment, accepts a vertices, adjacency, labels where:
    # vertices = list of NxC matrices where C is the same over all samples, N can be different between samples
    # adjacency = list of NxLxN tensors containing L NxN adjacency matrices of the given samples
    # labels = list of sample labels
    # len(vertices) == len(adjacency) == len(labels)
    def __init__(self, vertices=None, adjacency=None, labels=None, crop_if_possible=False):
        self.crop_if_possible = crop_if_possible
        self.graph_size = np.array([s.shape[0] for s in vertices]).astype(np.int64)

        self.largest_graph = max(self.graph_size)
        print_ext('Padding samples')
        self.graph_vertices = []
        self.graph_adjacency = []
        for i in range(len(vertices)):
            # pad all vertices to match size
            self.graph_vertices.append(np.pad(vertices[i].astype(np.float32), ((0, self.largest_graph-vertices[i].shape[0]), (0, 0)), 'constant', constant_values=(0)))

            # pad all adjacency matrices to match size
            self.graph_adjacency.append(np.pad(adjacency[i].astype(np.float32), ((0, self.largest_graph-adjacency[i].shape[0]), (0, 0), (0, self.largest_graph-adjacency[i].shape[0])), 'constant', constant_values=(0)))
            
        print_ext('Stacking samples')
        self.graph_vertices = np.stack(self.graph_vertices, axis=0)
        self.graph_adjacency = np.stack(self.graph_adjacency, axis=0)
        self.graph_labels = labels.astype(np.int64)
        
        self.no_samples = self.graph_labels.shape[0]
        
    # Create CV information
    def set_kfold(self, no_folds = 10, fold_id = 0):
        inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)
        self.fold_id = fold_id
        
        self.KFolds = list(inst.split(np.arange(self.no_samples)))
        self.train_idx, self.test_idx = self.KFolds[fold_id]
        self.no_samples_train = self.train_idx.shape[0]
        self.no_samples_test = self.test_idx.shape[0]
        print_ext('Data ready. no_samples_train:', self.no_samples_train, 'no_samples_test:', self.no_samples_test)
        
        self.train_batch_size = FLAGS.train_batch_size
        self.test_batch_size = FLAGS.test_batch_size
        if self.train_batch_size == 0:
            self.train_batch_size = self.no_samples_train
        if self.test_batch_size == 0:
            self.test_batch_size = self.no_samples_test
        self.train_batch_size = min(self.train_batch_size, self.no_samples_train)
        self.test_batch_size = min(self.test_batch_size, self.no_samples_test)
        
    # This function is cropped before batch
    # Slice each sample to improve performance
    def crop_single_sample(self, single_sample):
        vertices = tf.slice(single_sample[0], np.array([0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1]), tf.int64))
        vertices.set_shape([None, self.graph_vertices.shape[2]])
        adjacency = tf.slice(single_sample[1], np.array([0, 0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1, single_sample[3]]), tf.int64))
        adjacency.set_shape([None, self.graph_adjacency.shape[2], None])
        
        # V, A, labels, mask
        return [vertices, adjacency, single_sample[2], tf.expand_dims(tf.ones(tf.slice(tf.shape(vertices), [0], [1])), axis=-1)]

    # Create input_producers and batch queues
    def create_data(self, exp):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                # Create the training queue
                with tf.variable_scope('train_data') as scope:
                    print_ext('Creating training Tensorflow Tensors')
                    
                    # Create tensor with all training samples
                    training_samples = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
                    training_samples = [s[self.train_idx, ...] for s in training_samples]
                    
                    if self.crop_if_possible == False:
                        training_samples[3] = get_node_mask(training_samples[3], max_size=self.graph_vertices.shape[1])
                        
                    # Create tf.constants
                    training_samples = exp.create_input_variable(training_samples)
                    
                    # Slice first dimension to obtain samples
                    single_sample = tf.train.slice_input_producer(training_samples, shuffle=True, capacity=self.train_batch_size)
                    
                    # Cropping samples improves performance but is not required
                    if self.crop_if_possible:
                        print_ext('Cropping smaller graphs')
                        single_sample = self.crop_single_sample(single_sample)
                    
                    # creates training batch queue
                    train_queue = make_batch_queue(single_sample, capacity=self.train_batch_size*2, num_threads=6)

                # Create the test queue
                with tf.variable_scope('test_data') as scope:
                    print_ext('Creating test Tensorflow Tensors')
                    
                    # Create tensor with all test samples
                    test_samples = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
                    test_samples = [s[self.test_idx, ...] for s in test_samples]
                    
                    # If using mini-batch we will need a queue 
                    if self.test_batch_size != self.no_samples_test:
                        if self.crop_if_possible == False:
                            test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[1])
                        test_samples = exp.create_input_variable(test_samples)
                        
                        single_sample = tf.train.slice_input_producer(test_samples, shuffle=True, capacity=self.test_batch_size)
                        if self.crop_if_possible:
                            single_sample = self.crop_single_sample(single_sample)
                            
                        test_queue = make_batch_queue(single_sample, capacity=self.test_batch_size*2, num_threads=1)
                        
                    # If using full-batch no need for queues
                    else:
                        test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[1])
                        test_samples = exp.create_input_variable(test_samples)
                        
                # obtain batch depending on is_training and if test is a queue
                if self.test_batch_size == self.no_samples_test:
                    return tf.cond(exp.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_samples)
                return tf.cond(exp.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_queue.dequeue_many(self.test_batch_size))

