import graphcnn.setup.chemical as sc
from graphcnn.experiment_imagenet import *
from graphcnn.helper import *

class AlexNetExperiment():
    def create_network(self, net, input):
        net.create_network(input)
        net.make_cnn_layer(96, filter_size=11, stride=4, padding='VALID')
        net.make_pool_layer(padding='VALID')
        net.make_cnn_layer(256, filter_size=5)
        net.make_pool_layer(padding='VALID')
        net.make_cnn_layer(384, filter_size=3)
        net.make_cnn_layer(384, filter_size=3)
        net.make_cnn_layer(256, filter_size=3)
        net.make_pool_layer(padding='VALID')
        net.make_fc_layer(4096, with_bn=True)
        net.make_dropout_layer()
        net.make_fc_layer(4096, with_bn=True)
        net.make_dropout_layer()
        net.make_fc_layer(1000, with_act_func=False)
        
exp = GraphCNNImageNetExperiment('ImageNet', 'AlexNet', AlexNetExperiment())

# Add path to list of train/val files here
exp.train_list_file = 'list_of_train_files.txt'
exp.val_list_file = 'list_of_val_files.txt'

exp.num_iterations = 450000

exp.learning_rate_step = 100000
exp.starter_learning_rate = 0.01
exp.learning_rate_exp = 0.1

exp.train_batch_size = 256
exp.test_batch_size = 50

exp.iterations_per_test = 1000
acc, std = exp.run()