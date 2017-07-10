from .input_pipeline import *
from .experiment import GraphCNNExperiment
from .network import GraphCNNNetwork
from .layers import make_variable, make_bias_variable
class SingleGraphInputPipeline(InputPipeline):
    def __init__(self, vertices=None, adjacency=None, labels=None):
        self.largest_graph = vertices.shape[0]
        self.graph_size = [self.largest_graph]
        
        self.graph_vertices = np.expand_dims(vertices.astype(np.float32), axis=0)
        self.graph_adjacency = np.expand_dims(adjacency.astype(np.float32), axis=0)
        self.graph_labels = np.expand_dims(labels.astype(np.int64), axis=0)
        
        self.no_samples = self.graph_labels.shape[1]

    def set_kfold(self, no_folds = 10, fold_id = 0):
        inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)
        self.fold_id = fold_id
        
        self.KFolds = list(inst.split(np.arange(self.no_samples)))
        self.train_idx, self.test_idx = self.KFolds[fold_id]
        self.no_samples_train = self.train_idx.shape[0]
        self.no_samples_test = self.test_idx.shape[0]
        print_ext('Data ready. no_samples_train:', self.no_samples_train, 'no_samples_test:', self.no_samples_test)
        
        self.train_batch_size = self.no_samples_train
        self.test_batch_size = self.no_samples_test

    def create_data(self, exp):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                exp.print_ext('Creating training Tensorflow Tensors')
                
                vertices = self.graph_vertices[:, self.train_idx, :]
                adjacency = self.graph_adjacency[:, self.train_idx, :, :]
                adjacency = adjacency[:, :, :, self.train_idx]
                labels = self.graph_labels[:, self.train_idx]
                input_mask = np.ones([1, len(self.train_idx), 1]).astype(np.float32)
                
                train_input = [vertices, adjacency, labels, input_mask]
                train_input = exp.create_input_variable(train_input)
                
                vertices = self.graph_vertices
                adjacency = self.graph_adjacency
                labels = self.graph_labels
                
                input_mask = np.zeros([1, self.largest_graph, 1]).astype(np.float32)
                input_mask[:, self.test_idx, :] = 1
                test_input = [vertices, adjacency, labels, input_mask]
                test_input = exp.create_input_variable(test_input)
                
                return tf.cond(exp.net.is_training, lambda: train_input, lambda: test_input)

# BatchNormalization during test follows same behavior as training
class SingleGraphGraphCNNNetwork(GraphCNNNetwork):
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

# SingleGraphCNNExperiment overloads GraphCNNExperiment to support single graph samples (e.g. Cora)
# Loss function requires a mask that selects samples to report accuracy on.
class SingleGraphCNNExperiment(GraphCNNExperiment):
    def __init__(self, *nargs, **kwargs):
        GraphCNNExperiment.__init__(self, *nargs, **kwargs)
        self.net = SingleGraphGraphCNNNetwork()
        
    def create_loss_function(self):
        self.print_ext('Creating loss function and summaries')
        
        with tf.variable_scope('loss') as scope:
            labels = tf.cast(self.net.labels, tf.int64)
        
            inv_sum = (1./tf.reduce_sum(self.net.current_mask))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.net.current_V, labels=labels)
            cross_entropy = tf.multiply(tf.squeeze(self.net.current_mask), tf.squeeze(cross_entropy))
            cross_entropy = tf.reduce_sum(cross_entropy)*inv_sum

            correct_prediction = tf.cast(tf.equal(tf.argmax(self.net.current_V, 2), labels), tf.float32)
            correct_prediction = tf.multiply(tf.squeeze(self.net.current_mask), tf.squeeze(correct_prediction))
            accuracy = tf.reduce_sum(correct_prediction)*inv_sum
            
            tf.add_to_collection('losses', cross_entropy)
            tf.summary.scalar('loss', cross_entropy)
            
            self.max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
            self.max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
            
            max_acc = tf.cond(self.net.is_training, lambda: tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, accuracy)), lambda: tf.assign(self.max_acc_test, tf.maximum(self.max_acc_test, accuracy)))
            
            tf.summary.scalar('max_accuracy', max_acc)
            tf.summary.scalar('accuracy', accuracy)
            
            self.reports['accuracy'] = accuracy
            self.reports['max acc.'] = max_acc
            self.reports['cross_entropy'] = cross_entropy
