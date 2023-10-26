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
__global__ void MedusaQKVTransposeSplitKernel(T* medusa_k, 
                                              T* medusa_v,
                                              T* q_out,
                                              T* k_out,
                                              T* v_out,
                                              const T* qkv,
                                              const int* seq_lens_decoder,
                                              const int* padding_offsets,
                                              const int* cu_seqlens_k,
                                              int token_num,
                                              int num_head,
                                              int dim_head,
                                              int medusa_len,
                                              int seq_len) {
    // 1. Transpose qkv from [token_num, 3, num_head, dim_head] to [token_num, num_head, dim_head]  2 * [k_tokens_num, num_head, dim_head]
    // 2. Store to medusa_k/v [num_decoders, numhead, medusa_len, dim_head]

    const int token_idx = blockIdx.x;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    int seq_len_decoder = seq_lens_decoder[ori_bi];
    if (seq_len_decoder == 0) return;
    const int k_token_idx = ori_bi == 0 ? seq_len_decoder : cu_seqlens_k[ori_bi-1] + seq_len_decoder;


    const int hidden_size = num_head * dim_head;
    using InVec = AlignedVector<T, VecSize>;

    int decoder_id = -1;
    for (int i = 0; i <= ori_bi; i++) {
        decoder_id += (seq_len_decoder != 0);
    }

    InVec in_vec;
    for (int idx = threadIdx.x * VecSize; idx < 3 * num_head * dim_head; idx += blockDim.x * VecSize) {
        int qkv_id = idx / hidden_size;
        int h_id = (idx % (num_head * dim_head)) / dim_head;
        int h_offset = idx % dim_head;
        Load<T, VecSize>(qkv + token_idx * 3 * hidden_size + idx, &in_vec);
        if (qkv_id == 0) {
            Store<T, VecSize>(in_vec, q_out + token_idx * hidden_size + h_id * dim_head + h_offset);
        } else if (qkv_id == 1) {
            Store<T, VecSize>(in_vec, k_out + k_token_idx * hidden_size + h_id * dim_head + h_offset);  
            Store<T, VecSize>(in_vec, medusa_k + decoder_id * num_head * medusa_len * dim_head + h_id * medusa_len * dim_head + (ori_token_idx % seq_len) * dim_head + h_offset);  
        } else {
            Store<T, VecSize>(in_vec, v_out + k_token_idx * hidden_size + h_id * dim_head + h_offset);
            Store<T, VecSize>(in_vec, medusa_v + decoder_id * num_head * medusa_len * dim_head + h_id * medusa_len * dim_head + (ori_token_idx % seq_len) * dim_head + h_offset);
        }
    }
}

template<typename T, int VecSize>
__global__ void MedusaFetchCacheKVKernel(
                                        T* k_out, // [k_token_num, num_head, dim_head]
                                        T* v_out,
                                        T* cache_k, // [max_block_nums, numhead, block_size, dim_head]
                                        T* cache_v, 
                                        const int* block_tables, // [batch_size, pre_max_block_num]
                                        const int* seq_lens_decoder,
                                        const int* cu_seqlens_k,
                                        int num_head,
                                        int dim_head,
                                        int pre_max_block_num,
                                        int block_size) {
    const int b_id = blockIdx.x;
    const int base_id = cu_seqlens_k[b_id];
    const int seq_len = seq_lens_decoder[b_id];


    using InVec = AlignedVector<T, VecSize>;

    InVec k_vec;
    InVec v_vec;

    if (seq_len == 0) {
        int src_len = cu_seqlens_k[b_id+1] - base_id;
        for (int idx = threadIdx.x * VecSize; idx < num_head * src_len  * dim_head; idx += blockDim.x * VecSize) {
            int h_id = idx / (src_len * dim_head);
            int local_token_id = (idx % (src_len * dim_head)) / dim_head;
            int h_offset = idx % dim_head;

            int block_table_offset = local_token_id / block_size;
            int block_offset = local_token_id % block_size;
            int block_idx = *(block_tables + b_id * pre_max_block_num + block_table_offset);

            Load<T, VecSize>(k_out + (base_id + local_token_id) * num_head * dim_head + h_id * dim_head + h_offset, &k_vec);
            Load<T, VecSize>(v_out + (base_id + local_token_id) * num_head * dim_head + h_id * dim_head + h_offset, &v_vec);

            Store<T, VecSize>(k_vec, cache_k + 
                            block_idx * num_head * block_size * dim_head + 
                            h_id * block_size * dim_head +
                            block_offset * dim_head + 
                            h_offset);

            Store<T, VecSize>(v_vec, cache_v + 
                            block_idx * num_head * block_size * dim_head + 
                            h_id * block_size * dim_head +
                            block_offset * dim_head + 
                            h_offset);
        }
    } else {
        for (int idx = threadIdx.x * VecSize; idx < num_head * seq_len * dim_head; idx += blockDim.x * VecSize) {

            int h_id = idx / (seq_len * dim_head);
            int local_token_id = (idx % (seq_len * dim_head)) / dim_head;
            int h_offset = idx % dim_head;


            int block_table_offset = local_token_id / block_size;
            int block_offset = local_token_id % block_size;
            int block_idx = *(block_tables + b_id * pre_max_block_num + block_table_offset);

            Load<T, VecSize>(cache_k + 
                            block_idx * num_head * block_size * dim_head + 
                            h_id * block_size * dim_head +
                            block_offset * dim_head + 
                            h_offset, 
                            &k_vec);

            Load<T, VecSize>(cache_v + 
                            block_idx * num_head * block_size * dim_head + 
                            h_id * block_size * dim_head +
                            block_offset * dim_head + 
                            h_offset, 
                            &v_vec);


            Store<T, VecSize>(k_vec, k_out + (base_id + local_token_id) * num_head * dim_head + h_id * dim_head + h_offset);
            Store<T, VecSize>(v_vec, v_out + (base_id + local_token_id) * num_head * dim_head + h_id * dim_head + h_offset);
        }
    }

    
    
}

template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchMedusaQKVTransposeSplit(const paddle::Tensor& medusa_k,
                                                              const paddle::Tensor& medusa_v,
                                                              const paddle::Tensor& qkv,
                                                              const paddle::Tensor& block_tables,
                                                              const paddle::Tensor& cache_k,
                                                              const paddle::Tensor& cache_v,
                                                              const paddle::Tensor& seq_lens_encoder,
                                                              const paddle::Tensor& seq_lens_decoder,
                                                              const paddle::Tensor& cu_seqlens_q,
                                                              const paddle::Tensor& cu_seqlens_k,
                                                              const paddle::Tensor& padding_offsets,
                                                              const paddle::Tensor& input_ids,
                                                              int layer_id) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    auto cu_stream = qkv.stream();
    std::vector<int64_t> qkv_shape = qkv.shape();
    const int token_num = qkv_shape[0];
    const int bsz = seq_lens_encoder.shape()[0];
    const int num_head = medusa_k.shape()[2];
    const int dim_head = medusa_k.shape()[4];
    const int medusa_len = medusa_k.shape()[3];


    const int max_block_nums = cache_k.shape()[1]; 
    const int block_size = cache_k.shape()[3];
    const int pre_max_block_num = block_tables.shape()[1];


    const int max_seq_len = input_ids.shape()[1];

    int k_token_num = cu_seqlens_k.copy_to(paddle::CPUPlace(), true).data<int>()[bsz];

    auto q_out = paddle::full({token_num, num_head, dim_head}, 0, qkv.dtype(), qkv.place());
    auto k_out = paddle::full({k_token_num, num_head, dim_head}, 0, qkv.dtype(), qkv.place());
    auto v_out = paddle::full({k_token_num, num_head, dim_head}, 0, qkv.dtype(), qkv.place());


    constexpr int PackSize = VEC_16B / sizeof(DataType_);

    MedusaQKVTransposeSplitKernel<DataType_, PackSize><<<token_num, 512, 0 , cu_stream>>>(reinterpret_cast<DataType_*>(const_cast<data_t*>(medusa_k.data<data_t>() + layer_id * bsz * num_head * medusa_len * dim_head)),
                                                                                       reinterpret_cast<DataType_*>(const_cast<data_t*>(medusa_v.data<data_t>() + layer_id * bsz * num_head * medusa_len * dim_head)),
                                                                                       reinterpret_cast<DataType_*>(q_out.data<data_t>()),
                                                                                       reinterpret_cast<DataType_*>(k_out.data<data_t>()),
                                                                                       reinterpret_cast<DataType_*>(v_out.data<data_t>()),
                                                                                       reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
                                                                                       seq_lens_decoder.data<int>(),
                                                                                       padding_offsets.data<int>(),
                                                                                       cu_seqlens_k.data<int>(),
                                                                                       token_num,
                                                                                       num_head,
                                                                                       dim_head,
                                                                                       medusa_len,
                                                                                       max_seq_len
                                                                                       );

    MedusaFetchCacheKVKernel<DataType_, PackSize><<<bsz, 512, 0, cu_stream>>>(reinterpret_cast<DataType_*>(k_out.data<data_t>()),
                                                                            reinterpret_cast<DataType_*>(v_out.data<data_t>()),
                                                                            reinterpret_cast< DataType_*>(const_cast<data_t*>(cache_k.data<data_t>() + layer_id * max_block_nums * num_head * block_size * dim_head)),
                                                                            reinterpret_cast< DataType_*>(const_cast<data_t*>(cache_v.data<data_t>() + layer_id * max_block_nums * num_head * block_size * dim_head)),
                                                                            block_tables.data<int>(),
                                                                            seq_lens_decoder.data<int>(),
                                                                            cu_seqlens_k.data<int>(),
                                                                            num_head,
                                                                            dim_head,
                                                                            pre_max_block_num,
                                                                            block_size
                                                                            );
    return {q_out, k_out, v_out};
}

std::vector<paddle::Tensor> MedusaQKVTransposeSplit(const paddle::Tensor& medusa_k,
                                                              const paddle::Tensor& medusa_v,
                                                              const paddle::Tensor& qkv,
                                                              const paddle::Tensor& block_tables,
                                                              const paddle::Tensor& cache_k,
                                                              const paddle::Tensor& cache_v,
                                                              const paddle::Tensor& seq_lens_encoder,
                                                              const paddle::Tensor& seq_lens_decoder,
                                                              const paddle::Tensor& cu_seqlens_q,
                                                              const paddle::Tensor& cu_seqlens_k,
                                                              const paddle::Tensor& padding_offsets,
                                                              const paddle::Tensor& input_ids,
                                                              int layer_id) {
    switch (qkv.dtype()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchMedusaQKVTransposeSplit<paddle::DataType::BFLOAT16>(
                medusa_k,
                medusa_v,
                qkv,
                block_tables,
                cache_k,
                cache_v,
                seq_lens_encoder,
                seq_lens_decoder,
                cu_seqlens_q,
                cu_seqlens_k,
                padding_offsets,
                input_ids,
                layer_id
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchMedusaQKVTransposeSplit<paddle::DataType::FLOAT16>(
                medusa_k,
                medusa_v,
                qkv,
                block_tables,
                cache_k,
                cache_v,
                seq_lens_encoder,
                seq_lens_decoder,
                cu_seqlens_q,
                cu_seqlens_k,
                padding_offsets,
                input_ids,
                layer_id
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchMedusaQKVTransposeSplit<paddle::DataType::FLOAT32>(
                medusa_k,
                medusa_v,
                qkv,
                block_tables,
                cache_k,
                cache_v,
                seq_lens_encoder,
                seq_lens_decoder,
                cu_seqlens_q,
                cu_seqlens_k,
                padding_offsets,
                input_ids,
                layer_id
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16, bfloat16 and float32 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> MedusaQKVTransposeSplitInferShape(
                                                              const std::vector<int64_t>& medusa_k_shape,
                                                              const std::vector<int64_t>& medusa_v_shape,
                                                              const std::vector<int64_t>& qkv_shape,
                                                              const std::vector<int64_t>& block_tables_shape,
                                                              const std::vector<int64_t>& cache_k_shape,
                                                              const std::vector<int64_t>& cache_v_shape,
                                                              const std::vector<int64_t>& seq_lens_encoder_shape,
                                                              const std::vector<int64_t>& seq_lens_decoder_shape,
                                                              const std::vector<int64_t>& cu_seqlens_q_shape,
                                                              const std::vector<int64_t>& cu_seqlens_k_shape,
                                                              const std::vector<int64_t>& padding_offsets_shape,
                                                              const std::vector<int64_t>& input_ids_shape,
                                                              int layer_id) {
    int64_t bsz = seq_lens_encoder_shape[0];
    int64_t num_head = medusa_k_shape[2];
    int64_t dim_head = medusa_k_shape[4];
    return {{-1, num_head, dim_head}, {-1, num_head, dim_head}, {-1, num_head, dim_head}};
}

std::vector<paddle::DataType> MedusaQKVTransposeSplitInferDtype(const paddle::DataType& medusa_k_dtype,
                                                        const paddle::DataType& medusa_v_dtype,
                                                        const paddle::DataType& qkv_dtype,
                                                        const paddle::DataType& block_table_dtype,
                                                        const paddle::DataType& cache_k_dtype,
                                                        const paddle::DataType& cache_v_dtype,
                                                        const paddle::DataType& seq_lens_encoder_dtype,
                                                        const paddle::DataType& seq_lens_decoder_dtype,
                                                        const paddle::DataType& cu_seqlens_q_dtype,
                                                        const paddle::DataType& cu_seqlens_k_dtype,
                                                        const paddle::DataType& padding_offset_dtype,
                                                        const paddle::DataType& input_ids_dtype) {
    return {qkv_dtype, qkv_dtype, qkv_dtype};
}


PD_BUILD_OP(medusa_qkv_transpose_split_fetch_concat)
    .Inputs({"medusa_k", "medusa_v", "qkv", "block_tables", "cache_k", "cache_v", "seq_lens_encoder", "seq_lens_decoder", "cu_seqlens_q", "cu_seqlens_k", "padding_offsets", "input_ids"})
    .Outputs({"q_out", "k_out", "v_out"})
    .Attrs({"layer_id: int"})
    .SetKernelFn(PD_KERNEL(MedusaQKVTransposeSplit))
    .SetInferShapeFn(PD_INFER_SHAPE(MedusaQKVTransposeSplitInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MedusaQKVTransposeSplitInferDtype));