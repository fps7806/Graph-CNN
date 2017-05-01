from graphcnn.helper import *
from graphcnn.network import *
from graphcnn.layers import *
from graphcnn.image_helper import *
import numpy as np
import tensorflow as tf
from graphcnn.experiment import GraphCNNExperiment, make_batch_queue
from tensorflow.examples.tutorials.mnist import input_data

class GraphCNNMNISTExperiment(GraphCNNExperiment):
    def __init__(self, dataset_name, model_name, net_constructor):
        GraphCNNExperiment.__init__(self, dataset_name, model_name, net_constructor)
        mnist = input_data.read_data_sets("datasets/")

        self.train_data = np.expand_dims(mnist.train._images, axis=-1)
        self.train_labels = mnist.train._labels

        self.test_data = np.expand_dims(mnist.validation._images, axis=-1)
        self.test_labels = mnist.validation._labels
        
        self.image_size = 28
        
    # Create input_producers and batch queues
    def create_data(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                if self.is_eval:
                    num_epochs = 1
                else:
                    num_epochs = None
                # Create the test queue
                with tf.variable_scope('test_data') as scope:
                    self.print_ext('Creating test Tensorflow Tensors')
                    
                    # Create tensor with all training samples
                    test_samples = [self.test_data.astype(np.float32), self.test_labels]
                        
                    # Create tf.constants
                    test_samples = self.create_input_variable(test_samples)
                    
                    # Slice first dimension to obtain samples
                    single_sample = tf.train.slice_input_producer(test_samples, shuffle=True, capacity=self.test_batch_size, num_epochs=num_epochs)
                    
                    # creates training batch queue
                    test_queue = make_batch_queue(single_sample, capacity=self.test_batch_size*2, num_threads=2)
            
                if self.is_eval:
                    # obtain batch depending on is_training
                    input = test_queue.dequeue_up_to(self.test_batch_size)
                else:
                    # Create the training queue
                    with tf.variable_scope('train_data') as scope:
                        self.print_ext('Creating training Tensorflow Tensors')
                        
                        # Create tensor with all training samples
                        training_samples = [self.train_data.astype(np.float32), self.train_labels]
                            
                        # Create tf.constants
                        training_samples = self.create_input_variable(training_samples)
                        
                        # Slice first dimension to obtain samples
                        single_sample = tf.train.slice_input_producer(training_samples, shuffle=True, capacity=self.train_batch_size)
                        
                        # creates training batch queue
                        train_queue = make_batch_queue(single_sample, capacity=self.train_batch_size*2, num_threads=2)
                            
                    # obtain batch depending on is_training
                    input = tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_queue.dequeue_many(self.test_batch_size))
        A = self.create_input_variable([create_image_adj(self.image_size)])[0]
        A = tf.tile(tf.expand_dims(A, 0), [tf.shape(input[0])[0], 1, 1, 1])
        return [input[0], A, input[1], None]
        
    def run(self):
        self.is_eval = False
        GraphCNNExperiment.run(self)
        self.is_eval = True
        return self.eval()
        
    # Create graph (input, network, loss)
    # start/end threads
    def eval(self):
        tf.reset_default_graph()
        self.variable_initialization = {}
        
        self.print_ext('Evaluating model "%s"!' % self.model_name)
        if hasattr(self, 'fold_id') and self.fold_id:
            self.snapshot_path = './snapshots/%s/%s/' % (self.dataset_name, self.model_name + '_fold%d' % self.fold_id)
        else:
            self.snapshot_path = './snapshots/%s/%s/' % (self.dataset_name, self.model_name)
        
        self.net.is_training = tf.placeholder(tf.bool, shape=())
        self.net.global_step = tf.Variable(0,name='global_step',trainable=False)
        
        input = self.create_data()
        tf_no_samples = tf.shape(input[0])[0]
        self.net_constructor.create_network(self.net, input)
        self.create_loss_function()
        
        self.print_ext('Preparing evaluation')
        loss = tf.add_n(tf.get_collection('losses'))
        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer(), self.variable_initialization)
            
            saver = tf.train.Saver()
            self.load_model(sess, saver)
        
            self.print_ext('Starting threads')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            self.print_ext('Starting evaluation. test_batch_size:', self.test_batch_size)
            wasKeyboardInterrupt = False
            accuracy = 0
            total = 0
            try:
                acc_total = 0
                while True:
                    no_samples, acc = sess.run([tf_no_samples, self.net.accuracy], feed_dict={self.net.is_training:0})
                    total += no_samples
                    acc_total += acc*no_samples
            except tf.errors.OutOfRangeError as err:
                self.print_ext('Epoch completed!')
                accuracy = acc_total/total
            except KeyboardInterrupt as err:
                self.print_ext('Evaluation interrupted at %d' % i)
                wasKeyboardInterrupt = True
                raisedEx = err
            finally:
                self.print_ext('Evaluation completed, starting cleanup!')
                coord.request_stop()
                coord.join(threads)
                self.print_ext('Cleanup completed!')
                if wasKeyboardInterrupt:
                    raise raisedEx
            
            return accuracy, total