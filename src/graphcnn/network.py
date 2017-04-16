from graphcnn.layers import *
from graphcnn.network_description import GraphCNNNetworkDescription

class GraphCNNNetwork(object):
    def __init__(self):
        self.current_V = None
        self.current_A = None
        self.current_mask = None
        self.labels = None
        self.network_debug = False
        
    def create_network(self, input):
        self.current_V = input[0]
        self.current_A = input[1]
        self.labels = input[2]
        self.current_mask = input[3]
        
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
        
    def make_dropout_layer(self, keep_prob=0.5):
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
            self.current_V, self.current_A = make_graph_embed_pooling(self.current_V, self.current_A, mask=self.current_mask, no_vertices=no_vertices)
            self.current_mask = None
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