import graphcnn.setup.cora as sc
from graphcnn.experiment import *

dataset = sc.load_cora_dataset()

class CoraExperiment(SingleGraphCNNExperiment):
    def __init__(self):
        SingleGraphCNNExperiment.__init__(self, 'Cora', 'cora')
        self.num_iterations = 1000
        self.optimizer = 'adam'
        self.debug = True
        self.silent = True
        
    def create_network(self, input):
        SingleGraphCNNExperiment.create_network(self, input)
        self.make_embedding_layer(256)
        self.make_dropout_layer()
        
        self.make_graphcnn_layer(48)
        self.make_dropout_layer()
        self.make_embedding_layer(32)
        self.make_dropout_layer()
        
        
        self.make_graphcnn_layer(48)
        self.make_dropout_layer()
        self.make_embedding_layer(32)
        self.make_dropout_layer()
        
        input[0] =  self.make_graphcnn_layer(7, name='final', with_bn=False, with_act_func = False)
        return input
        
exp = CoraExperiment()
exp.preprocess_data(dataset)

acc, std = exp.run_kfold_experiments(no_folds=10)
print_ext('10-fold: %.2f (+- %.2f)' % (acc, std))