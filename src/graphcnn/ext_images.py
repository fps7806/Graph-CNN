from .input_pipeline import InputPipeline
from .helper import *
from .flags import FLAGS

class ImagesInputPipeline(InputPipeline):
    def __init__(self, train_data, train_labels, test_data, test_labels, image_size):
        self.train_data = train_data
        self.train_labels = train_labels

        self.test_data = test_data
        self.test_labels = test_labels

        self.no_samples_train = self.train_data.shape[0]
        self.no_samples_test = self.test_data.shape[0]

        self.image_size = image_size
        self.train_batch_size = FLAGS.train_batch_size
        self.test_batch_size = FLAGS.test_batch_size
        print_ext('Data ready. no_samples_train:', self.no_samples_train, 'no_samples_test:', self.no_samples_test)

    def create_image_adj(self, size):
        A = np.zeros([size*size, 8, size*size], np.float32)
        for i in range(size):
            for j in range(size):
                if i > 0 and j > 0: #up-left
                    A[i*size+j, 0, (i-1)*size+j-1] = 1
                    
                if i > 0: #up
                    A[i*size+j, 1, (i-1)*size+j] = 1
                    
                if i > 0 and j < size - 1: #up-right
                    A[i*size+j, 2, (i-1)*size+j+1] = 1
                    
                    
                if j > 0:#left
                    A[i*size+j, 3, i*size+j-1] = 1
                    
                if j < size - 1:#right
                    A[i*size+j, 4, i*size+j+1] = 1
                    
                    
                if i < size-1 and j > 0: #bottom-left
                    A[i*size+j, 5, (i+1)*size+j-1] = 1
                    
                if i < size-1:#bottom
                    A[i*size+j, 6, (i+1)*size+j] = 1
                    
                if i < size-1 and j < size - 1: # bottom right
                    A[i*size+j, 7, (i+1)*size+j+1] = 1
        return A

    def create_evaluation_data(self, exp):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                # Create the test queue
                exp.print_ext('Creating test Tensorflow Tensors')
                
                # Create tensor with all training samples
                test_samples = [self.test_data.astype(np.float32), self.test_labels]
                    
                # Create tf.constants
                test_samples = exp.create_input_variable(test_samples)
                
                # Slice first dimension to obtain samples
                single_sample = tf.train.slice_input_producer(test_samples, shuffle=True, capacity=self.test_batch_size, num_epochs=1)
                
                # creates training batch queue
                test_queue = make_batch_queue(single_sample, capacity=self.test_batch_size*2, num_threads=2)
            
                input = test_queue.dequeue_up_to(self.test_batch_size)
        A = exp.create_input_variable([self.create_image_adj(self.image_size)])[0]
        A = tf.tile(tf.expand_dims(A, 0), [tf.shape(input[0])[0], 1, 1, 1])
        return [input[0], A, input[1], None]
        
    # Create input_producers and batch queues
    def create_data(self, exp):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                # Create the test queue
                with tf.variable_scope('test_data') as scope:
                    exp.print_ext('Creating test Tensorflow Tensors')
                    
                    # Create tensor with all training samples
                    test_samples = [self.test_data.astype(np.float32), self.test_labels]
                        
                    # Create tf.constants
                    test_samples = exp.create_input_variable(test_samples)
                    
                    # Slice first dimension to obtain samples
                    single_sample = tf.train.slice_input_producer(test_samples, shuffle=True, capacity=self.test_batch_size, num_epochs=1)
                    
                    # creates training batch queue
                    test_queue = make_batch_queue(single_sample, capacity=self.test_batch_size*2, num_threads=2)
            
                # Create the training queue
                with tf.variable_scope('train_data') as scope:
                    exp.print_ext('Creating training Tensorflow Tensors')
                    
                    # Create tensor with all training samples
                    training_samples = [self.train_data.astype(np.float32), self.train_labels]
                        
                    # Create tf.constants
                    training_samples = exp.create_input_variable(training_samples)
                    
                    # Slice first dimension to obtain samples
                    single_sample = tf.train.slice_input_producer(training_samples, shuffle=True, capacity=self.train_batch_size)
                    
                    # creates training batch queue
                    train_queue = make_batch_queue(single_sample, capacity=self.train_batch_size*2, num_threads=2)
                        
                    # obtain batch depending on is_training
                input = tf.cond(exp.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_queue.dequeue_many(self.test_batch_size))
        A = exp.create_input_variable([self.create_image_adj(self.image_size)])[0]
        A = tf.tile(tf.expand_dims(A, 0), [tf.shape(input[0])[0], 1, 1, 1])
        return [input[0], A, input[1], None]