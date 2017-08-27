import tensorflow as tf
import numpy as np
import time
from scipy.sparse import random
import os
from graphcnn.sparse_helper import GraphCNNSparseAdjacency
from tensorflow.python.framework import ops

here = os.path.dirname(__file__)
if os.path.isfile(os.path.join(here, 'graphcnn_conv_sparse.so')):
    _graphcnn_conv_sparse_module = tf.load_op_library(os.path.join(here, 'graphcnn_conv_sparse.so'))

    def GraphConvolution(V, A):
        if isinstance(A, tf.Tensor):
            no_A = A.get_shape()[2].value
            no_features = V.get_shape()[2].value

            A_shape = tf.shape(A)
            A_reshape = tf.reshape(A, tf.stack([-1, A_shape[1]*no_A, A_shape[1]]))
            n = tf.matmul(A_reshape, V)
            return tf.reshape(n, [-1, A_shape[1], no_A, no_features])
        elif isinstance(A, GraphCNNSparseAdjacency):
            return _graphcnn_conv_sparse_module.graph_conv_sparse(V, A.indices, A.values, no_edge_features=A.no_A)
        else:
            raise NotImplementedError()

    @ops.RegisterGradient("GraphConvSparse")
    def _graph_conv_sparse_gradient(op, grad):
        return [_graphcnn_conv_sparse_module.graph_conv_sparse_gradient(grad, op.inputs[1], op.inputs[2]), None, None]

else:
    def GraphConvolution(V, A):
        no_A = A.get_shape()[2].value
        no_features = V.get_shape()[2].value

        A_shape = tf.shape(A)
        A_reshape = tf.reshape(A, tf.stack([-1, A_shape[1]*no_A, A_shape[1]]))
        n = tf.matmul(A_reshape, V)
        return tf.reshape(n, [-1, A_shape[1], no_A, no_features])