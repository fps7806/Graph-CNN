from .flags import *
from graphcnn.helper import *
from graphcnn.network import *
from graphcnn.layers import *
import numpy as np
import tensorflow as tf
import glob
import time

# This class is responsible for setting up and running experiments
# Also provides helper functions related to experiments (e.g. get accuracy)
class GraphCNNExperiment(object):
    def __init__(self, dataset_name, model_name, net_constructor):
        FLAGS.dataset_name = dataset_name
        # Initialize all defaults
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.crop_if_possible = True
        self.reports = {}
        
        self.net_constructor = net_constructor
        self.net = GraphCNNNetwork()
        self.net_desc = GraphCNNNetworkDescription()
        tf.reset_default_graph()
        
    # print_ext can be disabled through the silent flag
    def print_ext(self, *args):
        if FLAGS.silent == False:
            print_ext(*args)
            
    # Will retrieve the value stored as the maximum test accuracy on a trained network
    # SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES
    def get_max_accuracy(self):
        tf.reset_default_graph()
        self.net.global_step = tf.Variable(0,name='global_step',trainable=False)
        with tf.variable_scope('loss') as scope:
            max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            max_it = self.load_model(sess, saver)
            return sess.run(max_acc_test), max_it
        
    # Run all folds in a CV and calculate mean/std
    def run_kfold_experiments(self):
        no_folds = FLAGS.NO_FOLDS
        acc = []
        
        self.net_constructor.create_network(self.net_desc, [])
        desc = self.net_desc.get_description()
        self.print_ext('Running CV for:', desc)
        start_time = time.time()
        for i in range(no_folds):
            self.input.set_kfold(no_folds=no_folds, fold_id=i)
            cur_max, max_it = self.train()
            self.print_ext('Fold %d max accuracy: %g at %d' % (i, cur_max, max_it))
            acc.append(cur_max)
        acc = np.array(acc)
        mean_acc= np.mean(acc)*100
        std_acc = np.std(acc)*100
        self.print_ext('Result is: %.2f (+- %.2f)' % (mean_acc, std_acc))
        
        result_file_name = get_regex_flag('RESULTS_FILE')
        verify_dir_exists(result_file_name)
        with open(result_file_name, 'a+') as file:
            file.write('%s\t%s\t%d-fold\t%d seconds\t%.2f (+- %.2f)\n' % (str(datetime.now()), desc, no_folds, time.time()-start_time, mean_acc, std_acc))
        return mean_acc, std_acc
        
    def create_input_variable(self, input):
        for i in range(len(input)):
            placeholder = tf.placeholder(tf.as_dtype(input[i].dtype), shape=input[i].shape)
            var = tf.Variable(placeholder, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            self.variable_initialization[placeholder] = input[i]
            input[i] = var
        return input
     
    # Function called with the output of the Graph-CNN model
    # Should add the loss to the 'losses' collection and add any summaries needed (e.g. accuracy) 
    def create_loss_function(self):
        with tf.variable_scope('loss') as scope:
            labels = tf.cast(self.net.labels, tf.int64)
            self.print_ext('Creating loss function and summaries')
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.net.current_V, labels=labels))

            correct_prediction = tf.cast(tf.equal(tf.argmax(self.net.current_V, 1), labels), tf.float32)
            self.net.accuracy = tf.reduce_mean(correct_prediction)
            
            # we have 2 variables that will keep track of the best accuracy obtained in training/testing batch
            # SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES
            self.max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
            self.max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
            max_acc = tf.cond(self.net.is_training, lambda: tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, self.net.accuracy)), lambda: tf.assign(self.max_acc_test, tf.maximum(self.max_acc_test, self.net.accuracy)))
            
            tf.add_to_collection('losses', cross_entropy)
            tf.summary.scalar('accuracy', self.net.accuracy)
            tf.summary.scalar('max_accuracy', max_acc)
            tf.summary.scalar('cross_entropy', cross_entropy)
            
            # if silent == false display these statistics:
            self.reports['accuracy'] = self.net.accuracy
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
        int(latest[len(self.snapshot_path + 'model-'):])
        self.print_ext("Model restored at %d." % self.net.global_step.eval())
        return self.net.global_step.eval()
        
    def save_model(self, sess, saver):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None or self.net.global_step.eval() > int(latest[len(self.snapshot_path + 'model-'):]):
            self.print_ext('Saving model at %d' % self.net.global_step.eval())
            verify_dir_exists(self.snapshot_path)
            result = saver.save(sess, self.snapshot_path + 'model', global_step=self.net.global_step.eval())
            self.print_ext('Model saved to %s' % result)
      
    # Create graph (input, network, loss)
    # Handle checkpoints
    # Report summaries if silent == false
    # start/end threads
    def train(self):
        self.variable_initialization = {}
        
        self.print_ext('Training model "%s"!' % self.model_name)
        if hasattr(self.input, 'fold_id') and self.input.fold_id:
            self.snapshot_path = './snapshots/%s/%s/' % (self.dataset_name, self.model_name + '_fold%d' % self.input.fold_id)
            self.test_summary_path = './summary/%s/test/%s_fold%d' %(self.dataset_name, self.model_name, self.input.fold_id)
            self.train_summary_path = './summary/%s/train/%s_fold%d' %(self.dataset_name, self.model_name, self.input.fold_id)
        else:
            self.snapshot_path = './snapshots/%s/%s/' % (self.dataset_name, self.model_name)
            self.test_summary_path = './summary/%s/test/%s' %(self.dataset_name, self.model_name)
            self.train_summary_path = './summary/%s/train/%s' %(self.dataset_name, self.model_name)
        if FLAGS.save_checkpoints == False:
            i = 0
        else:
            i = self.check_model_iteration()
        tf.reset_default_graph()
        if i < FLAGS.num_iterations:
            self.print_ext('Creating training network')
            
            self.net.is_training = tf.placeholder(tf.bool, shape=())
            self.net.global_step = tf.Variable(0,name='global_step',trainable=False)
            
            
            input = self.input.create_data(self)
            self.net_constructor.create_network(self.net, input)
            self.create_loss_function()
            
            self.print_ext('Preparing training')
            loss = tf.add_n(tf.get_collection('losses'))
            if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
                loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            
            
            with tf.control_dependencies(update_ops):
                if FLAGS.optimizer == 'adam':
                    train_step = tf.train.AdamOptimizer().minimize(loss, global_step=self.net.global_step)
                else:
                    self.learning_rate = tf.train.exponential_decay(FLAGS.starter_learning_rate, self.net.global_step, FLAGS.learning_rate_step, FLAGS.learning_rate_exp, staircase=True)
                    train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(loss, global_step=self.net.global_step)
                    self.reports['lr'] = self.learning_rate
                    tf.summary.scalar('learning_rate', self.learning_rate)
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer(), self.variable_initialization)
                
                if FLAGS.save_checkpoints:
                    saver = tf.train.Saver()
                    self.load_model(sess, saver)
                                
                if FLAGS.summary_save:
                    self.print_ext('Starting summaries')
                    test_writer = tf.summary.FileWriter(self.test_summary_path, sess.graph)
                    train_writer = tf.summary.FileWriter(self.train_summary_path, sess.graph)
            
                summary_merged = tf.summary.merge_all()
            
                self.print_ext('Starting threads')
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                self.print_ext('Starting training. train_batch_size:', self.input.train_batch_size, 'test_batch_size:', self.input.test_batch_size)
                wasKeyboardInterrupt = False
                try:
                    total_training = 0.0
                    total_testing = 0.0
                    total_summary = 0.0
                    start_at = time.time()
                    while self.net.global_step.eval() < FLAGS.num_iterations:
                        if self.net.global_step.eval() % FLAGS.snapshot_iter == 0 and FLAGS.save_checkpoints:
                            self.save_model(sess, saver)
                            
                        if FLAGS.summary_save and (self.net.global_step.eval() % FLAGS.summary_period) == 0:
                            start_temp = time.time()
                            summary = sess.run(summary_merged, feed_dict={self.net.is_training:1})
                            train_writer.add_summary(summary, self.net.global_step.eval())
                            
                            summary = sess.run(summary_merged, feed_dict={self.net.is_training:0})
                            test_writer.add_summary(summary, self.net.global_step.eval())
                            
                            total_summary += time.time() - start_temp
                        
                        if self.net.global_step.eval() % FLAGS.iterations_per_test == 0:
                            start_temp = time.time()
                            reports = sess.run(self.reports, feed_dict={self.net.is_training:0})
                            total_testing += time.time() - start_temp
                            self.print_ext('Test Step %d Finished' % self.net.global_step.eval())
                            for key, value in reports.items():
                                self.print_ext('Test Step %d "%s" = ' % (self.net.global_step.eval(), key), value)
                                
                        start_temp = time.time()
                        reports, _ = sess.run([self.reports, train_step], feed_dict={self.net.is_training:1})
                        total_training += time.time() - start_temp
                        if self.net.global_step.eval() % FLAGS.display_iter == 0:
                            self.print_ext('Training Step %d Finished Timing (Train: %g, Test: %g, Summary: %g) after %g seconds' % (self.net.global_step.eval(), total_training, total_testing, total_summary, time.time()-start_at)) 
                            for key, value in reports.items():
                                self.print_ext('Training Step %d "%s" = ' % (self.net.global_step.eval(), key), value)

                except KeyboardInterrupt as err:
                    self.print_ext('Training interrupted at %d' % i)
                    wasKeyboardInterrupt = True
                    raisedEx = err
                finally:
                    if FLAGS.save_checkpoints:
                        self.save_model(sess, saver)
                    self.print_ext('Training completed, starting cleanup!')
                    coord.request_stop()
                    coord.join(threads)
                    self.print_ext('Cleanup completed!')
                    if wasKeyboardInterrupt:
                        raise raisedEx
                
                return sess.run([self.max_acc_test, self.net.global_step])
        else:
            self.print_ext('Model "%s" already trained!' % self.model_name)
            return self.get_max_accuracy()

    # Create graph (input, network, loss)
    # start/end threads
    def evaluate(self):
        tf.reset_default_graph()
        self.variable_initialization = {}
        
        self.print_ext('Evaluating model "%s"!' % self.model_name)
        if hasattr(self, 'fold_id') and self.fold_id:
            self.snapshot_path = './snapshots/%s/%s/' % (self.dataset_name, self.model_name + '_fold%d' % self.fold_id)
        else:
            self.snapshot_path = './snapshots/%s/%s/' % (self.dataset_name, self.model_name)
        
        self.net.is_training = tf.placeholder(tf.bool, shape=())
        self.net.global_step = tf.Variable(0,name='global_step',trainable=False)
        
        input = self.input.create_evaluation_data(self)
        tf_no_samples = tf.shape(input[0])[0]
        self.net_constructor.create_network(self.net, input)
        self.create_loss_function()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer(), self.variable_initialization)
            
            saver = tf.train.Saver()
            self.load_model(sess, saver)
        
            self.print_ext('Starting threads')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            self.print_ext('Starting evaluation. test_batch_size:', self.input.test_batch_size)
            wasKeyboardInterrupt = False
            accuracy = 0
            total = 0
            try:
                acc_total = 0
                while total < self.input.no_samples_test and not coord.should_stop():
                    no_samples, acc = sess.run([tf_no_samples, self.net.accuracy], feed_dict={self.net.is_training:0})
                    total += no_samples
                    acc_total += acc*no_samples
                self.print_ext('Epoch completed!')
                accuracy = acc_total/total
            except KeyboardInterrupt as err:
                self.print_ext('Evaluation interrupted at %d samples' % total)
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
