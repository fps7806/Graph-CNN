import tensorflow as tf

tf.app.flags.DEFINE_integer('num_iterations', 1500, "Number of samples to process in a training batch.")

import graphcnn
from graphcnn import FLAGS

def main(argv=None):
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
            
    exp = graphcnn.SingleGraphCNNExperiment('Cora', 'cora', CoraExperiment())
    exp.input = graphcnn.setup.load_cora_dataset()
            
    acc, std = exp.run_kfold_experiments()
    graphcnn.print_ext('%d-fold: %.2f (+- %.2f)' % (FLAGS.NO_FOLDS, acc, std))

if __name__ == '__main__':
    tf.app.run()