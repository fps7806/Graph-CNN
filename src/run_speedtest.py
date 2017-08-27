import tensorflow as tf
import graphcnn
from graphcnn import FLAGS
from graphcnn.helper import *
import numpy as np
import time
import pdb

class SpeedExperiment(object):
    def __init__(self):
        tf.reset_default_graph()
        self.net = graphcnn.GraphCNNNetwork()
        self.net.is_training = tf.placeholder_with_default(tf.constant(False), shape=())
        self.crop_if_possible = False

    def create_input_variable(self, input):
        for i in range(len(input)):
            var = tf.Variable(tf.constant(input[i]), trainable=False)
            input[i] = var
        return input

    def make_queue(self, training_samples, batch_size):
        # Create tf.constants
        training_samples = self.create_input_variable(training_samples)
        
        # Slice first dimension to obtain samples
        single_sample = tf.train.slice_input_producer(training_samples, shuffle=True, capacity=batch_size*10)
        
        return tf.train.batch(single_sample, batch_size, num_threads=4, capacity=batch_size*100, dynamic_pad=True)

def main(argv=None):
    exp = SpeedExperiment()
    graphcnn_conv_module = tf.load_op_library('./src/graphcnn/ops/graphcnn_conv.so')
    graphcnn_conv_sparse_module = tf.load_op_library('./src/graphcnn/ops/graphcnn_conv_sparse.so')
    graphcnn_conv_sparse_module_temp = tf.load_op_library('./src/graphcnn/ops/graphcnn_conv_sparse_temp.so')

    megabyte_divider = 4./(1024*1024)
    data = graphcnn.setup.load_protein_dataset('NCI1');batch_size = 128
    #data = graphcnn.setup.load_cora_dataset();batch_size = 1
    data.set_kfold()

    print('Full adjancecy tensor is {:e} floats ({:.0f} MB)'.format(data.graph_adjacency.size, data.graph_adjacency.size*megabyte_divider))
    data_sparse = graphcnn.batched_sparse_from_dense(data.graph_adjacency)
    print('Sparse adjancecy tensor is {:e} floats ({:.0f} MB)'.format(
        np.sum([d.size for d in data_sparse]), 
        np.sum([d.size for d in data_sparse])*megabyte_divider))

    A = data.graph_adjacency
    no_vertices = A.shape[3]
    A = np.transpose(A, [0, 2, 1, 3])

    A_batches = graphcnn.batched_sparse_from_dense(A, batch_dims=2, index_type=np.int64)
    V, A, indices, values, indices2, values2 = exp.make_queue([data.graph_vertices[:, :, :32], data.graph_adjacency, data_sparse[0], data_sparse[1], A_batches[0], A_batches[1]], batch_size=batch_size)

    no_A = data.graph_adjacency.shape[2]
    no_features = 32

    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, tf.stack([-1, A_shape[1]*no_A, A_shape[1]]))
    n = tf.matmul(A_reshape, V)
    n = tf.reshape(n, [-1, A_shape[1], no_A, no_features])

    result = graphcnn_conv_module.graph_conv(V, A)

    sparse_result = graphcnn_conv_sparse_module.graph_conv_sparse(V, indices, values, no_edge_features=no_A)
    sparse_result_temp = graphcnn_conv_sparse_module_temp.graph_conv_sparse_temp(V, indices, values, no_edge_features=no_A)

    V, indices, values = [tf.unstack(v) for v in [V, indices2, values2]]
    A = [[tf.SparseTensor(indices=indices[i][j, ...],
                        values=values[i][j, ...],
                         dense_shape=[no_vertices, no_vertices]) for j in range(indices[i].shape[0])] for i in range(len(indices))]

    sparse2_result = tf.stack([tf.stack([tf.sparse_tensor_dense_matmul(A[i][j], V[i])
                    for j in range(len(A[i]))]) for i in range(len(indices))])
    print(sparse2_result)
    sparse2_result = tf.transpose(sparse2_result, [0, 2, 1, 3])
    print(sparse2_result)

    no_evals = 10000
    with tf.Session() as sess:
        print('Initializing')
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('Test1')
        a, b, c, d = sess.run([n, sparse_result, sparse2_result, sparse_result_temp])
        assert np.allclose(a, b, rtol=1e-02, atol=1e-02)
        assert np.allclose(a, c, rtol=1e-02, atol=1e-02)
        assert np.allclose(a, d, rtol=1e-02, atol=1e-02)

        print('Wait a few second to fill queue')
        time.sleep(10)
        print('Running full')
        start = time.time()
        for i in range(no_evals):
            pass#n.eval()
        print('Tensorflow matmuls %d evals finished in %.4f seconds' % (no_evals, time.time()-start))

        print('Wait a few second to fill queue')
        time.sleep(10)
        start = time.time()
        for i in range(no_evals):
            pass#sparse_result.eval()
        print('sparse graphcnn_conv %d evals finished in %.4f seconds' % (no_evals, time.time()-start))

        print('Wait a few second to fill queue')
        time.sleep(10)
        start = time.time()
        for i in range(no_evals):
            sparse_result_temp.eval()
        print('sparse temp graphcnn_conv %d evals finished in %.4f seconds' % (no_evals, time.time()-start))

        print('Wait a few second to fill queue')
        time.sleep(10)
        start = time.time()
        for i in range(no_evals):
            pass#sparse2_result.eval()
        print('sparse matmul %d evals finished in %.4f seconds' % (no_evals, time.time()-start))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()