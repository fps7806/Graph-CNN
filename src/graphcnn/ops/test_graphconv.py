import tensorflow as tf
import numpy as np
import time
from scipy.sparse import random
from ..sparse_helper import GraphCNNSparseAdjacency
from . import custom_ops as ops

class GraphConvolutionTest(tf.test.TestCase):
    def testGraphConvolution(self):
        with self.test_session():
            no_A = 2
            no_features = 32
            no_vertices = 32
            batch_size = 2

            no_evals = 2

            V = np.random.randn(batch_size, no_vertices, no_features).astype(np.float32)
            sparse_A=random(batch_size* no_vertices, no_A * no_vertices, density=0.01).astype(np.float32)
            A = sparse_A.A
            A = np.reshape(A, [batch_size, no_vertices, no_A, no_vertices])


            sparse_A = GraphCNNSparseAdjacency.from_dense(A)
            A = tf.constant(A, tf.float32)
            V = tf.constant(V, tf.float32)
            n = ops.GraphConvolution(V, A)

            sparse_result = ops.GraphConvolution(V, sparse_A)
            self.assertAllClose(n.eval(), sparse_result.eval(), rtol=1e-02, atol=1e-02)

            grad_n = np.random.randn(*n.eval().shape).astype(np.float32)
            grad_v1 = tf.gradients(n, V, grad_n)
            grad_v2 = tf.gradients(sparse_result, V, grad_n)

            self.assertAllClose(grad_v1[0].eval(), grad_v2[0].eval(), rtol=1e-02, atol=1e-02)

if __name__ == "__main__":
    tf.test.main()