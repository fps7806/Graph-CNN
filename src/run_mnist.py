import tensorflow as tf

tf.app.flags.DEFINE_integer('train_batch_size', 128, "Number of samples to process in a training batch.")
tf.app.flags.DEFINE_integer('test_batch_size', 128, "Number of samples to process in a training batch.")
tf.app.flags.DEFINE_integer('num_iterations', 1000, "Number of samples to process in a training batch.")

import graphcnn
from graphcnn import FLAGS

def main(argv=None):
    class NCI1Experiment(object):
        def create_network(self, net, input):
            net.create_network(input)
            net.current_V = tf.reshape(net.current_V, [-1, 28, 28, 1])
            net.make_cnn_layer(64)
            
            net.make_fc_layer(128)
            net.make_fc_layer(10, name='final', with_bn=False, with_act_func = False)
            
    exp = graphcnn.GraphCNNExperiment('MNIST', 'test', NCI1Experiment())
    exp.input = graphcnn.setup.load_mnist_dataset()
    exp.train()

    graphcnn.print_ext('Results: %.2f (+- %.2f)' % exp.evaluate())

if __name__ == '__main__':
    tf.app.run()