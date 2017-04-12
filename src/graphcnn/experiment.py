from graphcnn.helper import *
from graphcnn.network import GraphCNNNetwork
from graphcnn.layers import *
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
import glob
import time
from tensorflow.python.training import queue_runner

# This function is used to create tf.cond compatible tf.train.batch alternative
def _make_batch_queue(input, capacity, num_threads=1):
    queue = tf.PaddingFIFOQueue(capacity=capacity, dtypes=[s.dtype for s in input], shapes=[s.get_shape() for s in input])
    tf.summary.scalar("fraction_of_%d_full" % capacity,
           tf.cast(queue.size(), tf.float32) *
           (1. / capacity))
    enqueue_ops = [queue.enqueue(input)]*num_threads
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))
    return queue

# This class is responsible for setting up and running experiments
# Also provides helper functions related to experiments (e.g. get accuracy)
class GraphCNNExperiment(GraphCNNNetwork):
    def __init__(self, dataset_name, model_name):
        # Initialize all defaults
        GraphCNNNetwork.__init__(self)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_iterations = 200
        self.iterations_per_test = 5
        self.display_iter = 5
        self.snapshot_iter = 1000000
        self.train_batch_size = 0
        self.test_batch_size = 0
        self.crop_if_possible = True
        self.debug = False
        self.starter_learning_rate = 0.1
        self.learning_rate_exp = 0.1
        self.learning_rate_step = 1000
        self.reports = {}
        self.silent = False
        self.optimizer = 'momentum'
        tf.reset_default_graph()
        
    # print_ext can be disabled through the silent flag
    def print_ext(self, *args):
        if self.silent == False:
            print_ext(*args)
            
    # Will retrieve the value stored as the maximum test accuracy on a trained network
    # SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES
    def get_max_accuracy(self):
        tf.reset_default_graph()
        with tf.variable_scope('loss') as scope:
            max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            max_it = self.load_model(sess, saver)
            return sess.run(max_acc_test), max_it
        
    # Run all folds in a CV and calculate mean/std
    def run_kfold_experiments(self, dataset, no_folds=10):
        acc = []
        for i in range(no_folds):
            tf.reset_default_graph()
            self.set_dataset_cv(dataset, no_folds=no_folds, fold_id=i)
            cur_max, max_it = self.run()
            self.print_ext('Fold %d max accuracy: %g at %d' % (i, cur_max, max_it))
            acc.append(cur_max)
        acc = np.array(acc)
        mean_acc= np.mean(acc)*100
        std_acc = np.std(acc)*100
        self.print_ext('Result is: %.2f (+- %.2f)' % (mean_acc, std_acc))
        return mean_acc, std_acc
        
    # Prepares samples for experiment, accepts a list (vertices, adjacency, labels) where:
    # vertices = list of NxC matrices where C is the same over all samples, N can be different between samples
    # adjacency = list of NxLxN tensors containing L NxN adjacency matrices of the given samples
    # labels = list of sample labels
    # len(vertices) == len(adjacency) == len(labels)
    def set_dataset_cv(self, dataset, no_folds = 10, fold_id = 0):
        inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)
        self.fold_id = fold_id
        
        self.graph_size = np.array([s.shape[0] for s in dataset[0]]).astype(np.int64)
        self.largest_graph = max(self.graph_size)
        self.print_ext('Padding samples')
        self.graph_vertices = []
        self.graph_adjacency = []
        for i in range(len(dataset[0])):
            # pad all vertices to match size
            self.graph_vertices.append(np.pad(dataset[0][i].astype(np.float32), ((0, self.largest_graph-dataset[0][i].shape[0]), (0, 0)), 'constant', constant_values=(0)))

            # pad all adjacency matrices to match size
            self.graph_adjacency.append(np.pad(dataset[1][i].astype(np.float32), ((0, self.largest_graph-dataset[1][i].shape[0]), (0, 0), (0, self.largest_graph-dataset[1][i].shape[0])), 'constant', constant_values=(0)))
            
        self.print_ext('Stacking samples')
        self.graph_vertices = np.stack(self.graph_vertices, axis=0)
        self.graph_adjacency = np.stack(self.graph_adjacency, axis=0)
        self.graph_labels = dataset[2].astype(np.int64)
        
        self.KFolds = list(inst.split(self.graph_vertices))
        self.train_idx, self.test_idx = self.KFolds[fold_id]
        self.no_samples = self.graph_labels.shape[0]
        self.no_samples_train = self.train_idx.shape[0]
        self.no_samples_test = self.test_idx.shape[0]
        self.print_ext('Data ready. no_samples_train:', self.no_samples_train, 'no_samples_test:', self.no_samples_test)
        
        if self.train_batch_size == 0:
            self.train_batch_size = self.no_samples_train
        if self.test_batch_size == 0:
            self.test_batch_size = self.no_samples_test
        
    # This function is cropped before batch
    # Slice each sample to improve performance
    def crop_single_sample(self, single_sample):
        vertices = tf.slice(single_sample[0], np.array([0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1]), tf.int64))
        vertices.set_shape([None, self.graph_vertices.shape[2]])
        adjacency = tf.slice(single_sample[1], np.array([0, 0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1, single_sample[3]]), tf.int64))
        adjacency.set_shape([None, self.graph_adjacency.shape[2], None])
        
        # V, A, labels, mask, extra
        return [vertices, adjacency, single_sample[2], tf.expand_dims(tf.ones(tf.slice(tf.shape(vertices), [0], [1])), axis=-1)] + self.crop_preprocessed_data(single_sample[4:], single_sample[3])
        
    # Create input_producers and batch queues
    def create_data(self, cond):
        if self.debug:
            seed = 102
        else:
            seed = None
    
        with tf.device("/cpu:0"):
        
            # Create the training queue
            with tf.variable_scope('train_data') as scope:
                self.print_ext('Creating training Tensorflow Tensors')
                
                # Create tensor with all training samples
                input_vertices = self.graph_vertices[self.train_idx, ...]
                input_adjacency = self.graph_adjacency[self.train_idx, ...]
                input_labels = self.graph_labels[self.train_idx, ...]
                input_size = self.graph_size[self.train_idx]
                
                # preprocess_data, not used in current experiments.
                single_sample = [input_vertices, input_adjacency, input_labels, input_size]
                single_sample += self.preprocess_data(single_sample)
                if self.crop_if_possible == False:
                    single_sample[3] = get_node_mask(input_size)
                    
                # Create tf.constants
                single_sample = [tf.constant(s) for s in single_sample]
                
                # Slice first dimension to obtain samples
                single_sample = tf.train.slice_input_producer(single_sample, shuffle=True, capacity=self.train_batch_size, seed=seed)
                
                # Cropping samples improves performance but is not required
                if self.crop_if_possible:
                    self.print_ext('Cropping smaller graphs')
                    single_sample = self.crop_single_sample(single_sample)
                
                # creates training batch queue
                train_queue = _make_batch_queue(single_sample, capacity=self.train_batch_size*2, num_threads=6)

            # Create the test queue
            with tf.variable_scope('test_data') as scope:
                self.print_ext('Creating test Tensorflow Tensors')
                
                # Create tensor with all test samples
                input_vertices = self.graph_vertices[self.test_idx, ...]
                input_adjacency = self.graph_adjacency[self.test_idx, ...]
                input_labels = self.graph_labels[self.test_idx, ...]
                input_size = self.graph_size[self.test_idx]
                
                # If using mini-batch we will need a queue 
                if self.test_batch_size != self.no_samples_test:
                    single_sample = [input_vertices, input_adjacency, input_labels, input_size]
                    single_sample += self.preprocess_data(single_sample)
                    if self.crop_if_possible == False:
                        single_sample[3] = get_node_mask(input_size)
                    single_sample = [tf.constant(s) for s in single_sample]
                    
                    single_sample = tf.train.slice_input_producer(single_sample, shuffle=True, capacity=self.test_batch_size, seed=seed)
                    if self.crop_if_possible:
                        single_sample = self.crop_single_sample(single_sample)
                        
                    test_queue = _make_batch_queue(single_sample, capacity=self.test_batch_size*2, num_threads=1)
                    
                # If using full-batch no need for queues
                else:
                    input_mask = get_node_mask(input_size)
                    input_vertices = input_vertices[:, :input_mask.shape[1], ...]
                    input_adjacency = input_adjacency[:, :input_mask.shape[1], :, :input_mask.shape[1]]
                    
                    test_samples = [input_vertices, input_adjacency, input_labels, input_size]
                    test_samples += self.preprocess_data(test_samples)
                    
                    test_samples[3] = input_mask
                    test_samples = [tf.constant(s) for s in test_samples]
                    
            # obtain batch depending on is_training and if test is a queue
            if self.test_batch_size == self.no_samples_test:
                return tf.cond(cond, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_samples)
            return tf.cond(cond, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_queue.dequeue_many(self.test_batch_size))
     
    # Function called with the output of the Graph-CNN model
    # Should add the loss to the 'losses' collection and add any summaries needed (e.g. accuracy) 
    def create_loss_function(self, input, is_training):
        with tf.variable_scope('loss') as scope:
            self.print_ext('Creating loss function and summaries')
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=input[0], labels=input[2]))

            correct_prediction = tf.cast(tf.equal(tf.argmax(input[0], 1), input[2]), tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            
            # we have 2 variables that will keep track of the best accuracy obtained in training/testing batch
            # SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES
            self.max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
            self.max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
            max_acc = tf.cond(is_training, lambda: tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, accuracy)), lambda: tf.assign(self.max_acc_test, tf.maximum(self.max_acc_test, accuracy)))
            
            tf.add_to_collection('losses', cross_entropy)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('max_accuracy', max_acc)
            tf.summary.scalar('cross_entropy', cross_entropy)
            
            # if silent == false display these statistics:
            self.reports['accuracy'] = accuracy
            self.reports['max acc.'] = max_acc
            self.reports['cross_entropy'] = cross_entropy
        
    # check if the model has a saved iteration and return the latest iteration step
    def check_model_iteration(self):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None:
            return 0
        return int(latest[len(self.snapshot_path + 'model-'):])
        
    # load_model if any checkpoint exist
    def load_model(self, sess, saver, ):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None:
            return 0
        saver.restore(sess, latest)
        i = int(latest[len(self.snapshot_path + 'model-'):])
        self.print_ext("Model restored at %d." % i)
        return i
        
    def save_model(self, sess, saver, i):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None or i != int(latest[len(self.snapshot_path + 'model-'):]):
            self.print_ext('Saving model at %d' % i)
            verify_dir_exists(self.snapshot_path)
            result = saver.save(sess, self.snapshot_path + 'model', global_step=i)
            self.print_ext('Model saved to %s' % result)
      
    # Create graph (input, network, loss)
    # Handle checkpoints
    # Report summaries if silent == false
    # start/end threads
    def run(self):
        self.print_ext('Training model "%s"!' % self.model_name)
        self.snapshot_path = './../snapshots/%s/%s/' % (self.dataset_name, self.model_name + '_fold%d' % self.fold_id)
        self.test_summary_path = './../summary/%s/test/%s_fold%d' %(self.dataset_name, self.model_name, self.fold_id)
        self.train_summary_path = './../summary/%s/train/%s_fold%d' %(self.dataset_name, self.model_name, self.fold_id)
        if self.debug:
            i = 0
        else:
            i = self.check_model_iteration()
        if i < self.num_iterations:
            self.print_ext('Creating training network')
            
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.global_step = tf.Variable(0,name='global_step',trainable=False)
            
            
            input = self.create_data(self.is_training)
            train_network = self.create_network(input)
            self.create_loss_function(train_network, self.is_training)
            
            self.print_ext('Preparing training')
            loss = tf.add_n(tf.get_collection('losses'))
            if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
                loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                
            self.reports['loss'] = loss
            tf.summary.scalar('loss', loss)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            
            
            with tf.control_dependencies(update_ops):
                if self.optimizer == 'adam':
                    train_step = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)
                else:
                    self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, self.learning_rate_step, self.learning_rate_exp, staircase=True)
                    train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(loss, global_step=self.global_step)
                    self.reports['lr'] = self.learning_rate
                    tf.summary.scalar('learning_rate', self.learning_rate)
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                if self.debug == False:
                    saver = tf.train.Saver()
                    self.load_model(sess, saver)
                                
                    self.print_ext('Starting summaries')
                    test_writer = tf.summary.FileWriter(self.test_summary_path, sess.graph)
                    train_writer = tf.summary.FileWriter(self.train_summary_path, sess.graph)
            
                train_merged = tf.summary.merge_all()
                test_merged = tf.summary.merge_all()
            
                self.print_ext('Starting threads')
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                self.print_ext('Starting training. train_batch_size:', self.train_batch_size, 'test_batch_size:', self.test_batch_size)
                wasKeyboardInterrupt = False
                try:
                    total_training = 0.0
                    total_testing = 0.0
                    start_at = time.time()
                    last_summary = time.time()
                    while i < self.num_iterations:
                        if i % self.snapshot_iter == 0 and self.debug == False:
                            self.save_model(sess, saver, i)
                        if i % self.iterations_per_test == 0:
                            start_temp = time.time()
                            summary, reports = sess.run([test_merged, self.reports], feed_dict={self.is_training:0})
                            total_testing += time.time() - start_temp
                            self.print_ext('Test Step %d Finished' % i)
                            for key, value in reports.items():
                                self.print_ext('Test Step %d "%s" = ' % (i, key), value)
                            if self.debug == False:
                                test_writer.add_summary(summary, i)
                            
                        start_temp = time.time()
                        summary, _, reports = sess.run([train_merged, train_step, self.reports], feed_dict={self.is_training:1})
                        total_training += time.time() - start_temp
                        i += 1
                        if ((i-1) % self.display_iter) == 0:
                            if self.debug == False:
                                train_writer.add_summary(summary, i-1)
                            total = time.time() - start_at
                            self.print_ext('Training Step %d Finished Timing (Training: %g, Test: %g) after %g seconds' % (i-1, total_training/total, total_testing/total, time.time()-last_summary)) 
                            for key, value in reports.items():
                                self.print_ext('Training Step %d "%s" = ' % (i-1, key), value)
                            last_summary = time.time()            
                        if (i-1) % 100 == 0:
                            total_training = 0.0
                            total_testing = 0.0
                            start_at = time.time()
                    if i % self.iterations_per_test == 0:
                        summary = sess.run(test_merged, feed_dict={self.is_training:0})
                        if self.debug == False:
                            test_writer.add_summary(summary, i)
                        self.print_ext('Test Step %d Finished' % i)
                except KeyboardInterrupt as err:
                    self.print_ext('Training interrupted at %d' % i)
                    wasKeyboardInterrupt = True
                    raisedEx = err
                finally:
                    if i > 0 and self.debug == False:
                        self.save_model(sess, saver, i)
                    self.print_ext('Training completed, starting cleanup!')
                    coord.request_stop()
                    coord.join(threads)
                    self.print_ext('Cleanup completed!')
                    if wasKeyboardInterrupt:
                        raise raisedEx
                
                return sess.run([self.max_acc_test, self.global_step])
        else:
            self.print_ext('Model "%s" already trained!' % self.model_name)
            return self.get_max_accuracy()

# SingleGraphCNNExperiment overloads GraphCNNExperiment to support single graph samples (e.g. Cora)
# BatchNormalization during test follows same behavior as training
# Loss function requires a mask that selects samples to report accuracy on.
class SingleGraphCNNExperiment(GraphCNNExperiment):
    def set_dataset_cv(self, dataset, no_folds = 10, fold_id = 0):
        inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)
        self.fold_id = fold_id
        self.KFolds = list(inst.split(dataset[0]))
        self.train_idx, self.test_idx = self.KFolds[fold_id]
        
        self.graph_size = dataset[0].shape[0]
        
        self.graph_vertices = np.expand_dims(dataset[0].astype(np.float32), axis=0)
        self.graph_adjacency = np.expand_dims(dataset[1].astype(np.float32), axis=0)
        self.graph_labels = np.expand_dims(dataset[2].astype(np.int64), axis=0)
        
        self.no_samples_train = self.train_idx.shape[0]
        self.no_samples_test = self.test_idx.shape[0]
        self.print_ext('Data ready. no_samples_train:', self.no_samples_train, 'no_samples_test:', self.no_samples_test)
        
        if self.train_batch_size == 0:
            self.train_batch_size = self.no_samples_train
        if self.test_batch_size == 0:
            self.test_batch_size = self.no_samples_test
            
            
    def make_batchnorm_layer(self, name=None):
        axis = -1
        with tf.variable_scope(name, default_name='BatchNorm') as scope:
            input_size = self.current_V.get_shape()[axis].value
            if axis == -1:
                axis = len(self.current_V.get_shape())-1
            axis_arr = [i for i in range(len(self.current_V.get_shape())) if i != axis]
            batch_mean, batch_var = tf.nn.moments(self.current_V, axis_arr)
            
            gamma = make_variable('gamma', input_size, initializer=tf.constant_initializer(1))
            beta = make_bias_variable('bias', input_size)
            self.current_V = tf.nn.batch_normalization(self.current_V, batch_mean, batch_var, beta, gamma, 1e-3)
            return self.current_V
    def create_data(self, cond, batch_size=1):
        self.print_ext('Creating training Tensorflow Tensors')
        
        vertices = tf.constant(self.graph_vertices[:, self.train_idx, :])
        adjacency = self.graph_adjacency[:, self.train_idx, :, :]
        adjacency = tf.constant(adjacency[:, :, :, self.train_idx])
        labels = tf.constant(self.graph_labels[:, self.train_idx])
        input_mask = np.ones([1, len(self.train_idx), 1]).astype(np.float32)
        
        train_input = [vertices, adjacency, labels, tf.constant(input_mask)]
        
        vertices = tf.constant(self.graph_vertices)
        adjacency = tf.constant(self.graph_adjacency)
        labels = tf.constant(self.graph_labels)
        
        input_mask = np.zeros([1, self.graph_size, 1]).astype(np.float32)
        input_mask[:, self.test_idx, :] = 1
        test_input = [vertices, adjacency, labels, tf.constant(input_mask)]
        
        return tf.cond(cond, lambda: train_input, lambda: test_input)
        
    def create_loss_function(self, input, is_training):
        self.print_ext('Creating loss function and summaries')
        
        with tf.variable_scope('loss') as scope:
            inv_sum = (1./tf.reduce_sum(input[3]))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=input[0], labels=input[2])
            cross_entropy = tf.multiply(tf.squeeze(input[3]), tf.squeeze(cross_entropy))
            cross_entropy = tf.reduce_sum(cross_entropy)*inv_sum

            correct_prediction = tf.cast(tf.equal(tf.argmax(input[0], 2), input[2]), tf.float32)
            correct_prediction = tf.multiply(tf.squeeze(input[3]), tf.squeeze(correct_prediction))
            accuracy = tf.reduce_sum(correct_prediction)*inv_sum
            
            tf.add_to_collection('losses', cross_entropy)
            tf.summary.scalar('loss', cross_entropy)
            
            self.max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
            self.max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
            
            max_acc = tf.cond(is_training, lambda: tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, accuracy)), lambda: tf.assign(self.max_acc_test, tf.maximum(self.max_acc_test, accuracy)))
            
            tf.summary.scalar('max_accuracy', max_acc)
            tf.summary.scalar('accuracy', accuracy)
            
            self.reports['accuracy'] = accuracy
            self.reports['max acc.'] = max_acc
            self.reports['cross_entropy'] = cross_entropy
