// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"

template<typename T>
__global__ void MedusaGatherKernel(T* result, // [num_decoders, num_posterior, self.medusa]
                             const T* prob, // [num_decoders, num_posterior, self.medusa, vocab_size]
                             const int64_t* index, // [num_decoders, num_posterior, self.medusa]
                             int64_t num_decoders,
                             int64_t num_posterior,
                             int64_t num_medusa,
                             int64_t vocab_size
                             ) {
    for (int64_t idx = blockIdx.x * gridDim.x + threadIdx.x; idx < num_decoders * num_posterior * num_medusa; idx += blockDim.x * gridDim.x) {
        int64_t decoder_id = idx / (num_posterior * num_medusa);
        int64_t posterior_id = (idx % (num_posterior * num_medusa)) / num_medusa;
        int64_t medusa_id = idx % num_medusa;

        int64_t index_value = index[idx];
        result[idx] = prob[idx * vocab_size + index_value];
    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchMedusaGather(const paddle::Tensor& prob, const paddle::Tensor& index) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;    

    auto cu_stream = prob.stream();

    auto num_decoders = prob.shape()[0];
    auto num_posterior = prob.shape()[1];
    auto num_medusa = prob.shape()[2];
    auto vocab_size = prob.shape()[3];

    auto result = paddle::empty({num_decoders, num_posterior, num_medusa}, prob.dtype(), prob.place());

    constexpr int grid_size = 108;
    constexpr int block_size = 512;

    MedusaGatherKernel<DataType_><<<grid_size,block_size, 0,  cu_stream>>>(
                                                                    reinterpret_cast<DataType_*>(result.data<data_t>()),
                                                                    reinterpret_cast<DataType_*>(const_cast<data_t*>(prob.data<data_t>())),
                                                                    index.data<int64_t>(),
                                                                    num_decoders,
                                                                    num_posterior,
                                                                    num_medusa,
                                                                    vocab_size
                                                                        );
}

std::vector<paddle::Tensor> MedusaGather(const paddle::Tensor& prob, const paddle::Tensor& index) {
    switch (prob.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchMedusaGather<paddle::DataType::BFLOAT16>(
                prob, index
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchMedusaGather<paddle::DataType::FLOAT16>(
                prob, index
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchMedusaGather<paddle::DataType::FLOAT32>(
                prob, index
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


std::vector<std::vector<int64_t>> MedusaGatherInferShape(
                                                            const std::vector<int64_t>& prob_shape,
                                                            const std::vector<int64_t>& index_shape) {
    return {index_shape};
}

std::vector<paddle::DataType> MedusaGatherInferDtype(
                                                        const paddle::DataType& prob_dtype,
                                                            const paddle::DataType& index_dtype) {
    return {prob_dtype};
}

PD_BUILD_OP(medusa_gather)
    .Inputs({"prob", "index"})
    .Outputs({"res"})
    .SetKernelFn(PD_KERNEL(MedusaGather));