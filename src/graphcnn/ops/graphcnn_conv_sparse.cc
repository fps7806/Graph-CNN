#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

#include <iostream>

using namespace std;

REGISTER_OP("GraphConvSparse")
    .Input("vertices: float32")
    .Input("adjacency_indices: int32")
    .Input("adjacency_values: float32")
    .Attr("no_edge_features: int")
    .Output("conv: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ShapeHandle v_shape;
      ShapeHandle idx_shape;
      ShapeHandle values_shape;

      // Validate shapes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &v_shape)); // V
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &idx_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &values_shape));

      // Batch dims match between inputs.
      ShapeHandle v_batch_dims;
      ShapeHandle a_batch_dims;
      ShapeHandle batch_dims;
      TF_RETURN_IF_ERROR(c->Subshape(idx_shape, 0, 0, &a_batch_dims));
      TF_RETURN_IF_ERROR(c->Subshape(v_shape, 0, 0, &v_batch_dims));
      TF_RETURN_IF_ERROR(c->Merge(a_batch_dims, v_batch_dims, &batch_dims));

      ShapeHandle out = c->MakeShape({c->Dim(v_shape, 0), c->Dim(v_shape, 1), 2, c->Dim(v_shape, 2)});

      c->set_output(0, out);
      return Status::OK();
    });

class GraphConvSparse : public OpKernel {
 public:
  explicit GraphConvSparse(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("no_edge_features", &_no_edge_features));
    OP_REQUIRES(context, _no_edge_features >= 0,
                errors::InvalidArgument("Need no_edge_features >= 0, got ",
                                        _no_edge_features));

  }

  void Compute(OpKernelContext* context) override {
    // TODO verify DIMENSIONS
    // Grab the input tensor
    const Tensor& vertices_tensor = context->input(0);
    const Tensor& idx_tensor = context->input(1);
    const Tensor& values_tensor = context->input(2);

    auto vertices = vertices_tensor.tensor<float, 3>();
    auto indices = idx_tensor.tensor<int, 3>();
    auto values = values_tensor.tensor<float, 2>();

    // Create an output tensor
    TensorShape out_shape;
    out_shape.AddDim(vertices_tensor.dim_size(0));
    out_shape.AddDim(vertices_tensor.dim_size(1));
    out_shape.AddDim(_no_edge_features);
    out_shape.AddDim(vertices_tensor.dim_size(2));
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &output_tensor));
    auto output = output_tensor->tensor<float, 4>();
    auto output_flat = output_tensor->flat<float>();

    const int N = output_flat.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
    const int batch_size = vertices_tensor.dim_size(0);
    const int num_features = vertices_tensor.dim_size(2);
    const int num_threads = std::min(thread_pool->num_threads, batch_size);

    auto f = [&](int thread_id) {
        // Set all but the first element of the output tensor to 0.
        for(int b =thread_id; b < batch_size;b+=num_threads)
        {
          for(int i =0; i < idx_tensor.dim_size(1);++ i)
          {
            auto in_v = indices(b, i, 0);
            auto a = indices(b, i, 1);
            auto out_v = indices(b, i, 2);

            // Need to handle cases with odd number of features(?)
            auto outputMap = Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>(
                (float*)&output(b, in_v, a, 0), num_features);
            auto verticesMap = Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>, Eigen::Aligned>(
                (const float*)&vertices(b, out_v, 0), num_features);
            outputMap += verticesMap * values(b, i);
          }
        }
    };

    BlockingCounter counter(num_threads-1);
    for (int i = 1; i < num_threads; ++i) {
        thread_pool->workers->Schedule([&, i]() {
            f(i);
            counter.DecrementCount();
        });
    }
    f(0);
    counter.Wait();
  }
 private:
  int _no_edge_features;
};


REGISTER_KERNEL_BUILDER(Name("GraphConvSparse").Device(DEVICE_CPU), GraphConvSparse);



REGISTER_OP("GraphConvSparseGradient")
    .Input("gradient: float32")
    .Input("adjacency_indices: int32")
    .Input("adjacency_values: float32")
    .Output("conv: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ShapeHandle v_shape;
      ShapeHandle idx_shape;
      ShapeHandle values_shape;

      // Validate shapes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &v_shape)); // V
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &idx_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &values_shape));

      // Batch dims match between inputs.
      ShapeHandle v_batch_dims;
      ShapeHandle a_batch_dims;
      ShapeHandle batch_dims;
      TF_RETURN_IF_ERROR(c->Subshape(idx_shape, 0, 0, &a_batch_dims));
      TF_RETURN_IF_ERROR(c->Subshape(v_shape, 0, 0, &v_batch_dims));
      TF_RETURN_IF_ERROR(c->Merge(a_batch_dims, v_batch_dims, &batch_dims));

      ShapeHandle out = c->MakeShape({c->Dim(v_shape, 0), c->Dim(v_shape, 1), c->Dim(v_shape, 3)});

      c->set_output(0, out);
      return Status::OK();
    });

class GraphConvSparseGradient : public OpKernel {
 public:
  explicit GraphConvSparseGradient(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    // TODO verify DIMENSIONS
    // Grab the input tensor
    const Tensor& vertices_tensor = context->input(0);
    const Tensor& idx_tensor = context->input(1);
    const Tensor& values_tensor = context->input(2);

    auto vertices = vertices_tensor.tensor<float, 4>();
    auto indices = idx_tensor.tensor<int, 3>();
    auto values = values_tensor.tensor<float, 2>();

    // Create an output tensor
    TensorShape out_shape;
    out_shape.AddDim(vertices_tensor.dim_size(0));
    out_shape.AddDim(vertices_tensor.dim_size(1));
    out_shape.AddDim(vertices_tensor.dim_size(3));
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &output_tensor));
    auto output = output_tensor->tensor<float, 3>();
    auto output_flat = output_tensor->flat<float>();

    const int N = output_flat.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
    const int batch_size = vertices_tensor.dim_size(0);
    const int num_features = vertices_tensor.dim_size(3);
    const int num_threads = std::min(thread_pool->num_threads, batch_size);

    auto f = [&](int thread_id) {
        // Set all but the first element of the output tensor to 0.
        for(int b =thread_id; b < batch_size;b+=num_threads)
        {
          for(int i =0; i < idx_tensor.dim_size(1);++ i)
          {
            auto in_v = indices(b, i, 0);
            auto a = indices(b, i, 1);
            auto out_v = indices(b, i, 2);

            // Need to handle cases with odd number of features(?)
            auto outputMap = Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>(
                (float*)&output(b, out_v, 0), num_features);
            auto verticesMap = Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>, Eigen::Aligned>(
                (const float*)&vertices(b, in_v, a, 0), num_features);
            outputMap += verticesMap * values(b, i);
          }
        }
    };

    BlockingCounter counter(num_threads-1);
    for (int i = 1; i < num_threads; ++i) {
        thread_pool->workers->Schedule([&, i]() {
            f(i);
            counter.DecrementCount();
        });
    }
    f(0);
    counter.Wait();
  }
 private:
  int _no_edge_features;
};


REGISTER_KERNEL_BUILDER(Name("GraphConvSparseGradient").Device(DEVICE_CPU), GraphConvSparseGradient);
