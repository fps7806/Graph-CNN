chemical_datasets_list = ['DD', 'ENZYMES', 'MUTAG', 'NCI1', 'NCI109']

def load_protein_dataset(dataset_name):
    import scipy.io
    import numpy as np
    import datetime
    import graphcnn
    from .helper import locate_or_download_file, locate_or_extract_file, get_file_location
    
    if dataset_name not in chemical_datasets_list:
        print_ext('Dataset doesn\'t exist. Options:', chemical_datasets_list)
        return
    locate_or_download_file('proteins/proteins_data.zip', 'http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/data.zip')
    locate_or_extract_file('proteins/proteins_data.zip', 'proteins/data')
    mat = scipy.io.loadmat(get_file_location('proteins/data/%s.mat' % dataset_name))
    
    input = mat[dataset_name]
    labels = mat['l' + dataset_name.lower()]
    labels = labels - min(labels)
    
    node_labels = input['nl']
    v_labels = 0
    for i in range(node_labels.shape[1]):
        v_labels = max(v_labels, max(node_labels[0, i]['values'][0, 0])[0])
    
    e_labels = 1    
    edge_labels = input[0, 0]['el']
    for i in range(input.shape[1]):
        for j in range(input[0, i]['el']['values'][0, 0].shape[0]):
            if input[0, i]['el']['values'][0, 0][j, 0].shape[0] > 0:
                e_labels = max(e_labels, max(input[0, i]['el']['values'][0, 0][j, 0])[0])
    
    # For each sample
    samples_V = []
    samples_A = []
    max_no_nodes = 0
    for i in range(input.shape[1]):
        no_nodes = node_labels[0, i]['values'][0, 0].shape[0]
        max_no_nodes = max(max_no_nodes, no_nodes)
        V = np.ones([no_nodes, v_labels])
        for l in range(v_labels):
            V[..., l] = np.equal(node_labels[0, i]['values'][0, 0][..., 0], l+1).astype(np.float32)
        samples_V.append(V)
        A = np.zeros([no_nodes, e_labels, no_nodes])
        for j in range(no_nodes):
            for k in range(input[0, i]['al'][j, 0].shape[1]):
                A[j, input[0, i]['el']['values'][0, 0][j, 0][0, k]-1, input[0, i]['al'][j, 0][0, k]-1] = 1
        samples_A.append(A)

    dataset = graphcnn.InputPipeline(vertices = np.array(samples_V), adjacency=np.array(samples_A), labels=np.reshape(labels, [-1]))

    return dataset