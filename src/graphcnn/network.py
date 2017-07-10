from graphcnn.layers import *
from graphcnn.network_description import GraphCNNNetworkDescription
from graphcnn.visualization import *
import matplotlib.pyplot as plt
import numpy as np
        
import threading
drawing_lock = threading.Lock()

class GraphCNNNetwork(object):
    def __init__(self):
        self.current_V = None
        self.current_A = None
        self.current_mask = None
        self.labels = None
        self.network_debug = False
        self.current_pos = None
        
    def create_network(self, input):
        self.current_V = input[0]
        self.current_A = input[1]
        self.labels = input[2]
        self.current_mask = input[3]
        
        # Visual helper
        self.current_pos = None
        self.visual_indices = None
        
        if self.network_debug:
            size = tf.reduce_sum(self.current_mask, axis=1)
            self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), tf.reduce_max(size), tf.reduce_mean(size)], message='Input V Shape, Max size, Avg. Size:')
        
        return input
        
        
    def make_batchnorm_layer(self):
        self.current_V = make_bn(self.current_V, self.is_training, mask=self.current_mask, num_updates = self.global_step)
        return self.current_V
        
    # Equivalent to 0-hop filter
    def make_embedding_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Embed') as scope:
            self.current_V = make_embedding_layer(self.current_V, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V, self.current_A, self.current_mask
        
    def make_dropout_layer(self, keep_prob=0.5, name=None):
        with tf.variable_scope(name, default_name='Dropout') as scope:
            self.current_V = tf.cond(self.is_training, lambda:tf.nn.dropout(self.current_V, keep_prob=keep_prob), lambda:(self.current_V))
            return self.current_V
        
    def make_graphcnn_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Graph-CNN') as scope:
            self.current_V = make_graphcnn_layer(self.current_V, self.current_A, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape())-1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var], message='"%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V
        
    def make_graph_embed_pooling(self, no_vertices=1, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='GraphEmbedPool') as scope:
            self.current_V, self.current_A, P = make_graph_embed_pooling(self.current_V, self.current_A, mask=self.current_mask, no_vertices=no_vertices)
            self.current_mask = None
            
            if self.current_pos is not None:
                sliced_P = tf.slice(P, [self.visual_indices, 0, 0], [1, tf.shape(self.current_pos)[1], -1])
                self.current_pos = tf.matmul(sliced_P, self.current_pos, transpose_a=True)
            
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape())-1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var], message='Pool "%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V, self.current_A, self.current_mask
            
    def make_fc_layer(self, no_filters, name=None, with_bn=False, with_act_func=True):
        with tf.variable_scope(name, default_name='FC') as scope:
            self.current_mask = None
            
            if len(self.current_V.get_shape()) > 2:
                no_input_features = int(np.prod(self.current_V.get_shape()[1:]))
                self.current_V = tf.reshape(self.current_V, [-1, no_input_features])
            self.current_V = make_embedding_layer(self.current_V, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V
        
        
    def make_cnn_layer(self, no_filters, name=None, with_bn=False, with_act_func=True, filter_size=3, stride=1, padding='SAME'):
        with tf.variable_scope(None, default_name='conv') as scope:
            dim = self.current_V.get_shape()[-1]
            kernel = make_variable_with_weight_decay('weights',
                                                 shape=[filter_size, filter_size, dim, no_filters],
                                                 stddev=math.sqrt(1.0/(no_filters*filter_size*filter_size)),
                                                 wd=0.0005)
            conv = tf.nn.conv2d(self.current_V, kernel, [1, stride, stride, 1], padding=padding)
            biases = make_bias_variable('biases', [no_filters])
            self.current_V = tf.nn.bias_add(conv, biases)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            return self.current_V
            
    def make_pool_layer(self, padding='SAME'):
        with tf.variable_scope(None, default_name='pool') as scope:
            dim = self.current_V.get_shape()[-1]
            self.current_V = tf.nn.max_pool(self.current_V, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding, name=scope.name)

            return self.current_V
            
    def make_visual_layer(self, name):
        def _create_visual(A, V, step, is_training, pos=None):
            try:
                drawing_lock.acquire()
                fig = plt.figure(figsize=(8, 8))
                pos = display_graph(V, A, pos)
                fig.canvas.draw()
                ncols, nrows = fig.canvas.get_width_height()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(1, nrows, ncols, 3)
                if is_training:
                    fig.savefig('visual/train_%d_%s.png' % (step, name), dpi=fig.dpi)
                else:
                    fig.savefig('visual/test_%d_%s.png' % (step, name), dpi=fig.dpi)
                plt.close(fig)
            finally:
                drawing_lock.release()
            return img, np.expand_dims(pos.astype(np.float32), 0)
        
        self.visual_indices = 0
        if self.current_mask is not None:
            size = tf.cast(tf.reduce_sum(self.current_mask[self.visual_indices]), tf.int32)
        else:
            size = tf.shape(self.current_A)[1]
        verify_dir_exists('visual/')
        V = tf.slice(self.current_V[self.visual_indices], [0, 0], [size, -1])
        A = tf.slice(self.current_A[self.visual_indices], [0, 0, 0], [size, -1, size])
        
        input_array = [A, V, self.global_step, self.is_training]
        if self.current_pos is not None:
            pos = tf.slice(self.current_pos[self.visual_indices], [0, 0], [size, -1])
            input_array.append(pos)
        img, self.current_pos = tf.py_func(_create_visual, input_array, [tf.uint8, tf.float32]) 
        tf.summary.image(name, img)
        
