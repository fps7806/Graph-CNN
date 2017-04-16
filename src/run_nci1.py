import graphcnn.setup.chemical as sc
from graphcnn.experiment import *

dataset = sc.load_protein_dataset('NCI1')

# Decay value for BatchNorm layers, seems to work better with 0.3
GraphCNNGlobal.BN_DECAY = 0.3
class NCI1Experiment(object):
    def create_network(self, net, input):
        net.create_network(input)
        net.make_graphcnn_layer(64)
        net.make_graphcnn_layer(64)
        net.make_graph_embed_pooling(no_vertices=32)
            
        net.make_graphcnn_layer(32)
        
        net.make_graph_embed_pooling(no_vertices=8)
            
        net.make_fc_layer(256)
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func = False)
        
exp = GraphCNNExperiment('NCI1', 'nci1', NCI1Experiment())

exp.num_iterations = 1500
exp.train_batch_size = 128
exp.optimizer = 'adam'
exp.debug = True

exp.preprocess_data(dataset)
acc, std = exp.run_kfold_experiments(no_folds=10)
print_ext('10-fold: %.2f (+- %.2f)' % (acc, std))