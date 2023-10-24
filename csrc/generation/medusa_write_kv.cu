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

template<typename T, int VecSize>
__global__ void MedusaWriteKVKernel(T* cache_k,
                               T* cache_v,
                               const T* medusa_k, 
                               const T* medusa_v,
                               const int* block_tables, 
                               const int* accept_length,
                               const int* seq_lens_decoder,
                               const int* insert_index_decoder,
                               int num_decoders,
                               int num_head,
                               int num_medusa,
                               int dim_head,
                               int pre_max_block_num,
                               int block_size,
                               int max_block_nums
                               ) {
    int layer_id = blockIdx.z;
    int decoder_id = blockIdx.y;
    int head_id = blockIdx.x;

    int b_id = insert_index_decoder[decoder_id];
    int seq_len = seq_lens_decoder[b_id];
    int this_len = accept_length[decoder_id];
    const int* block_table = block_tables + b_id * pre_max_block_num;

    using InVec = AlignedVector<T, VecSize>;

    InVec k_vec, v_vec;

    int medusa_base_id = layer_id * num_decoders * num_head * num_medusa * dim_head + decoder_id * num_head * num_medusa * dim_head + head_id * num_medusa * dim_head;
    int cache_base_id = layer_id * max_block_nums * num_head * block_size * dim_head;

    for (int idx = threadIdx.x * VecSize; idx < num_medusa * dim_head; idx += blockDim.x * VecSize) {
        int token_id = idx / dim_head;
        int offset = idx % dim_head;
        if (token_id < this_len) {
            int block_idx = (seq_len + token_id) / block_size;
            int block_offset = (seq_len + token_id) % block_size;
            int phisycal_block_id = block_table[block_idx];

            Load<T, VecSize>(medusa_k+medusa_base_id + token_id * dim_head + offset, &k_vec);
            Load<T, VecSize>(medusa_v+medusa_base_id + token_id * dim_head + offset, &v_vec);

            Store<T, VecSize>(k_vec, cache_k + cache_base_id + phisycal_block_id *  num_head * block_size * dim_head + head_id * block_size * dim_head + block_offset * dim_head + offset);
            Store<T, VecSize>(v_vec, cache_v + cache_base_id + phisycal_block_id *  num_head * block_size * dim_head + head_id * block_size * dim_head + block_offset * dim_head + offset);
        }
    }
}

template <paddle::DataType D>
void LaunchMedusaWriteKV(const paddle::Tensor& medusa_k, // [num_layers, num_decoders, numhead, self.medusa+1, dim_head]
              const paddle::Tensor& medusa_v, 
              const paddle::Tensor& cache_k, // [num_layers, max_block_nums, numhead, block_size, dim_head]
              const paddle::Tensor& cache_v,
              const paddle::Tensor& block_tables,// batch_size, pre_max_block_num
              const paddle::Tensor& accept_length,
              const paddle::Tensor& seq_lens_decoder,
              const paddle::Tensor& insert_index_decoder
              ) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;    

    auto cu_stream = medusa_k.stream();

    auto num_layers = medusa_k.shape()[0];
    auto num_decoders = medusa_k.shape()[1];
    auto num_head =  medusa_k.shape()[2];
    auto num_medusa = medusa_k.shape()[3];
    auto dim_head = medusa_k.shape()[4];
    auto max_block_nums = cache_k.shape()[1];
    auto block_size = cache_k.shape()[3];
    auto pre_max_block_num = block_tables.shape()[1];

    dim3 grid(num_head, num_decoders, num_layers);
    dim3 block(512);

    constexpr int VecSize = 16 / sizeof(data_t);


    MedusaWriteKVKernel<DataType_, VecSize><<<grid, block, 0, cu_stream>>>(reinterpret_cast<DataType_*>(const_cast<data_t*>(cache_k.data<data_t>())),
                                                                            reinterpret_cast<DataType_*>(const_cast<data_t*>(cache_v.data<data_t>())),
                                                                            reinterpret_cast<const DataType_*>(medusa_k.data<data_t>()),
                                                                            reinterpret_cast<const DataType_*>(medusa_v.data<data_t>()),
                                                                            block_tables.data<int>(),
                                                                            accept_length.data<int>(),
                                                                            seq_lens_decoder.data<int>(),
                                                                            insert_index_decoder.data<int>(),
                                                                            num_decoders, num_head, num_medusa, dim_head, pre_max_block_num, block_size, max_block_nums);
                                                                            
}

void MedusaWriteKV(const paddle::Tensor& medusa_k, // [num_layers, num_decoders, numhead, self.medusa+1, dim_head]
              const paddle::Tensor& medusa_v, 
              const paddle::Tensor& cache_k, // [num_layers, max_block_nums, numhead, block_size, dim_head]
              const paddle::Tensor& cache_v,
              const paddle::Tensor& block_tables,// batch_size, pre_max_block_num
              const paddle::Tensor& accept_length,
              const paddle::Tensor& seq_lens_decoder,
              const paddle::Tensor& insert_index_decoder) {
    switch (medusa_k.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchMedusaWriteKV<paddle::DataType::BFLOAT16>(
                medusa_k, medusa_v, cache_k, cache_v, block_tables, accept_length, seq_lens_decoder, insert_index_decoder
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchMedusaWriteKV<paddle::DataType::FLOAT16>(
                medusa_k, medusa_v, cache_k, cache_v, block_tables, accept_length, seq_lens_decoder, insert_index_decoder
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchMedusaWriteKV<paddle::DataType::FLOAT32>(
                medusa_k, medusa_v, cache_k, cache_v, block_tables, accept_length, seq_lens_decoder, insert_index_decoder
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

std::vector<std::vector<int64_t>> MedusaWriteKVInferShape(
                                                            const std::vector<int64_t>& medusa_k_shape,
                                                            const std::vector<int64_t>& medusa_v_shape,
                                                            const std::vector<int64_t>& cache_k_shape,
                                                            const std::vector<int64_t>& cache_v_shape,
                                                            const std::vector<int64_t>& block_tables_shape,
                                                            const std::vector<int64_t>& accept_length_shape,
                                                            const std::vector<int64_t>& seq_lens_decoder_shape,
                                                            const std::vector<int64_t>& insert_index_decoder_shape) {
    return {cache_k_shape, cache_v_shape};
}

std::vector<paddle::DataType> MedusaWriteKVInferDtype(
                                                        const paddle::DataType& medusa_k_dtype,
                                                            const paddle::DataType& medusa_v_dtype,
                                                            const paddle::DataType& cache_k_dtype,
                                                            const paddle::DataType& cache_v_dtype,
                                                            const paddle::DataType& block_tables_dtype,
                                                            const paddle::DataType& accept_length_dtype,
                                                            const paddle::DataType& seq_lens_decoder_dtype,
                                                            const paddle::DataType& insert_index_decoder_dtype) {
    return {cache_k_dtype, cache_v_dtype};
}

PD_BUILD_OP(medusa_write_kv)
    .Inputs({"medusa_k", "medusa_v", "cache_k", "cache_v", "block_tables", "accept_length", "seq_lens_decoder", "insert_index_decoder"})
    .Outputs({"cache_k_out", "cache_v_out"})
    .SetInplaceMap({{"cache_k", "cache_k_out"}, {"cache_v", "cache_v_out"}})
    .SetKernelFn(PD_KERNEL(MedusaWriteKV));