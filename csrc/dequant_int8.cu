#include "helper.h"
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<stdio.h>
#include<algorithm>


constexpr int DequantKernelVecSize = 4;

template <typename data_t>
inline HOSTDEVICE data_t roundWithTiesToEven(data_t x) {
  data_t xLower = floor(x);
  data_t xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  data_t dLower = x - xLower;
  data_t dUpper = xUpper - x;
  return static_cast<data_t>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename data_t, int VecSize>
__global__ void DequantKernel(data_t* output,
                              const int32_t* input,
                              const int m,  // batch size
                              const int n,  // hidden
                              const float* dequant_out_scale_data) {
  int numel = m * n;
  int stride = blockDim.x * gridDim.x * VecSize;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int col_id = idx % n;

  AlignedVector<int32_t, VecSize> in_vec;
  AlignedVector<float, VecSize> out_scale_vec;
  AlignedVector<data_t, VecSize> out_vec;

  for (; idx < numel; idx += stride) {
    Load<int32_t, VecSize>(input + idx, &in_vec);
    Load<float, VecSize>(dequant_out_scale_data + col_id, &out_scale_vec);

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] =
          static_cast<data_t>(static_cast<float>(in_vec[i]) * out_scale_vec[i]);
    }

    Store<data_t, VecSize>(out_vec, output + idx);
  }
}
template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchDequantInt8(const paddle::Tensor& input,
                                              const paddle::Tensor& detype_input,
                                              const paddle::Tensor& scale) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    std::vector<int64_t> input_shape = input.shape();
    auto output=paddle::full(input_shape, 0, detype_input.dtype(), input.place());
    int64_t m = input_shape[0];
    int64_t n = input_shape[1];
    // printf("launch dequant int8 m:%d,n:%d \n", m,n);

    int64_t numel = m*n;
    constexpr int64_t thread_per_block = 512;
    int64_t block_per_grid = (numel / DequantKernelVecSize + thread_per_block - 1) / thread_per_block;
    auto stream = input.stream();
    DequantKernel<DataType_, DequantKernelVecSize>
        <<<block_per_grid, thread_per_block, 0, stream>>>(
            reinterpret_cast<DataType_*>(output.data<data_t>()),
            reinterpret_cast<const int32_t*>(input.data<int32_t>()), m, n, 
            reinterpret_cast<const float*>(scale.data<float>()));

    return {output};

}

std::vector<paddle::Tensor> DequantInt8(const paddle::Tensor& input,
                                        const paddle::Tensor& detype_input,
                                        const paddle::Tensor& out_scale
                                        ) {
    switch (detype_input.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchDequantInt8<paddle::DataType::BFLOAT16>(
                input, detype_input, out_scale
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchDequantInt8<paddle::DataType::FLOAT16>(
                input, detype_input, out_scale
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchDequantInt8<paddle::DataType::FLOAT32>(
                input, detype_input, out_scale

            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
        }
    }
}



std::vector<std::vector<int64_t>> DequantInt8Shape(const std::vector<int64_t>& input_shape) {
    return {input_shape};
}

std::vector<paddle::DataType> DequantInt8Dtype(const paddle::DataType& input_dtype, const paddle::DataType& dtype_input_dtype) {
    return {dtype_input_dtype};
}

PD_BUILD_OP(dequant_int8)
    .Inputs({"intput","dtype_input","out_scale"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(DequantInt8))
    .SetInferShapeFn(PD_INFER_SHAPE(DequantInt8Shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DequantInt8Dtype));