import graphcnn.setup.cora as sc
from graphcnn.experiment import *

dataset = sc.load_cora_dataset()

class CoraExperiment():
    def create_network(self, net, input):
        net.create_network(input)
        net.make_embedding_layer(256)
        net.make_dropout_layer()
        
        net.make_graphcnn_layer(48)
        net.make_dropout_layer()
        net.make_embedding_layer(32)
        net.make_dropout_layer()
        
        
        net.make_graphcnn_layer(48)
        net.make_dropout_layer()
        net.make_embedding_layer(32)
        net.make_dropout_layer()
        
        net.make_graphcnn_layer(7, name='final', with_bn=False, with_act_func = False)
        
exp = SingleGraphCNNExperiment('Cora', 'cora', CoraExperiment())

exp.num_iterations = 1000
exp.optimizer = 'adam'
exp.debug = True
        
exp.preprocess_data(dataset)

acc, std = exp.run_kfold_experiments(no_folds=10)
print_ext('10-fold: %.2f (+- %.2f)' % (acc, std))