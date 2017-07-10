def load_mnist_dataset():
    from graphcnn import ImagesInputPipeline
    from tensorflow.examples.tutorials.mnist import input_data
    import numpy as np

    mnist = input_data.read_data_sets("datasets/")

    train_data = np.expand_dims(mnist.train._images, axis=-1)
    train_labels = mnist.train._labels

    test_data = np.expand_dims(mnist.validation._images, axis=-1)
    test_labels = mnist.validation._labels

    return ImagesInputPipeline(train_data, train_labels, test_data, test_labels, 28)
