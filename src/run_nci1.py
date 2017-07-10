import tensorflow as tf

tf.app.flags.DEFINE_float('BN_DECAY', 0.3, "Normalization decay used for BatchNorm.")
tf.app.flags.DEFINE_integer('train_batch_size', 128, "Number of samples to process in a training batch.")
tf.app.flags.DEFINE_integer('num_iterations', 1500, "Number of samples to process in a training batch.")

import graphcnn
from graphcnn import FLAGS

def main(argv=None):

    # Decay value for BatchNorm layers, seems to work better with 0.3
    class NCI1Experiment(object):
        def create_network(self, net, input):
            net.create_network(input)
            net.make_visual_layer('InputGraph')
            net.make_graphcnn_layer(64)
            net.make_graphcnn_layer(64)
            net.make_graph_embed_pooling(no_vertices=32)
            net.make_visual_layer('Pool1')
                
            net.make_graphcnn_layer(32)
            
            net.make_graph_embed_pooling(no_vertices=8)
            net.make_visual_layer('Pool2')
                
            net.make_fc_layer(256)
            net.make_fc_layer(2, name='final', with_bn=False, with_act_func = False)
            
    exp = graphcnn.GraphCNNExperiment('NCI1', 'nci1', NCI1Experiment())
    exp.input = graphcnn.setup.load_protein_dataset('NCI1')

    acc, std = exp.run_kfold_experiments()
    graphcnn.print_ext('%d-fold: %.2f (+- %.2f)' % (FLAGS.NO_FOLDS, acc, std))

if __name__ == '__main__':
    tf.app.run()