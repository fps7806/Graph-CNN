
class GraphCNNNetworkDescription(object):
    def __init__(self):
        self.network_description = []
        
    def create_network(self, input):
        self.network_description = []
        return input
        
    def get_description(self):
        return '-'.join(self.network_description)
        
    # Default names
    def __getattr__(self, name):
        return lambda *x, **key_x: self.make_default_layer(name, x, key_x)
        
    def make_default_layer(self, name, x, key_x):
        if name.startswith('make_'):
            name = name[5:]
        if name.endswith('_layer'):
            name = name[:-6]
            
        name = name.upper()
        
        if len(x) > 0:
            name = name + '(' + ','.join([str(s) for s in x]) +')'
    
        self.add_layer_desc(name)
        
    def add_layer_desc(self, desc):
        self.network_description.append(desc)
        
    def make_batchnorm_layer(self):
        pass
        
    # Equivalent to 0-hop filter
    def make_embedding_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        self.add_layer_desc('Embed(%d)' % no_filters)
        
    def make_dropout_layer(self, keep_prob=0.5):
        self.add_layer_desc('Dropout(%.2f)' % keep_prob)
        
    def make_graphcnn_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        self.add_layer_desc('CNN(%d)' % no_filters)
        
    def make_graph_embed_pooling(self, no_vertices=1, name=None, with_bn=True, with_act_func=True):
        self.add_layer_desc('GEP(%d)' % no_vertices)
        