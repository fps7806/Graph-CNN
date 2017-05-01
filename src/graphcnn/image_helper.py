import numpy as np

def create_image_adj(size):
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