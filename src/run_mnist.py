import graphcnn.setup.chemical as sc
from graphcnn.experiment_mnist import *
from graphcnn.helper import *
from graphcnn.visualization import *
import matplotlib.pyplot as plt

class NCI1Experiment(object):
    def create_network(self, net, input):
        net.create_network(input)
        net.current_V = tf.reshape(net.current_V, [-1, 28, 28, 1])
        net.make_cnn_layer(64)
        
        net.make_fc_layer(128)
        net.make_fc_layer(10, name='final', with_bn=False, with_act_func = False)
        
exp = GraphCNNMNISTExperiment('MNIST', 'test', NCI1Experiment())

exp.num_iterations = 1000
exp.train_batch_size = 128
exp.test_batch_size = 128
exp.optimizer = 'adam'
exp.image_size = 28

acc, std = exp.run()
print_ext('10-fold: %.2f on %d samples' % (acc*100, std))