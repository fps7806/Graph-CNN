TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -shared src/graphcnn/ops/graphcnn_conv_sparse.cc -o src/graphcnn/ops/graphcnn_conv_sparse.so -fPIC -I $TF_INC -O2 -undefined dynamic_lookup
