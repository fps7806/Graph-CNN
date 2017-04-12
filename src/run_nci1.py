import graphcnn.setup.chemical as sc
from graphcnn.experiment import *
from graphcnn.helper import *

dataset = sc.load_protein_dataset('NCI1')

# Decay value for BatchNorm layers, seems to work better with 0.3
GraphCNNGlobal.BN_DECAY = 0.3
class NCI1Experiment(GraphCNNExperiment):
    def __init__(self):
        GraphCNNExperiment.__init__(self, 'NCI1', 'nci1')
        self.num_iterations = 1500
        self.train_batch_size = 128
        self.optimizer = 'adam'
        self.silent = True
        self.debug = True
        
    def create_network(self, input):
        GraphCNNExperiment.create_network(self, input)

        self.make_graphcnn_layer(64)
        self.make_graphcnn_layer(64)
        self.make_graph_embed_pooling(no_vertices=32)
            
        self.make_graphcnn_layer(32)
        
        self.make_graph_embed_pooling(no_vertices=8)
            
        self.make_fc_layer(256)
        input[0] = self.make_fc_layer(2, name='final', with_bn=False, with_act_func = False)
        return input
        
exp = NCI1Experiment()

acc, std = exp.run_kfold_experiments(dataset, no_folds=10)
print_ext('10-fold %.2f: %.2f (+- %.2f)' % (GraphCNNGlobal.BN_DECAY, acc, std))