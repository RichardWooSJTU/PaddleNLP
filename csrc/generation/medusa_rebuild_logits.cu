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
constexpr int32_t WARP_SIZE = 32; 
constexpr int32_t HALF_WARP = 16; 

template <typename T>
__inline__ __device__ T WarpReduceSum(T val, unsigned lane_mask) {
  #pragma unroll
  for (int mask = HALF_WARP; mask > 0; mask >>= 1){
    val +=  __shfl_xor_sync(lane_mask, val, mask, WARP_SIZE);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, unsigned mask) {
    static __shared__ T smem[WARP_SIZE];
    int32_t lane_id = threadIdx.x % WARP_SIZE;
    int32_t warp_id = threadIdx.x / WARP_SIZE;

    val = WarpReduceSum(val, mask);

    if (lane_id == 0) {
        smem[warp_id] = val;
    }

    __syncthreads();

    T abs_max_val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? smem[threadIdx.x] : static_cast<T>(0.0f);
    abs_max_val = WarpReduceSum(abs_max_val, mask);
    return abs_max_val;
}


__global__ void GetNumDecoder(int* result,
                            const int* seq_lens_decoder, 
                            int bsz) {
    int32_t decoder_num = 0;

    for (int32_t idx = threadIdx.x; idx < bsz; idx += blockDim.x) {
        decoder_num = static_cast<int32_t>(seq_lens_decoder[idx] != 0);
    }

    BlockReduceSum<int32_t>(decoder_num, 0xFFFFFFFF);
    if (threadIdx.x == 0) {
        result[0] = decoder_num;
    }

}

__global__ void MedusaRebuildLogitsKernel(int* logits_decoder_index, // num_decoders, medusa_len
                                          int* logits_encoder_index, // num_encoders
                                        int* insert_index_decoder, //[num_decoder]
                                        int* insert_index_encoder, //[num_encoder]
                                          const int* seq_lens_encoder, //[bsz]
                                          const int* seq_lens_decoder, //[bsz]
                                          const int* padding_offsets, // [token_num]
                                        //   const int* retrieve_indices, // [num_posterior, self.medusa+1]
                                          int token_num,
                                          int seq_len,
                                          int medusa_len
                                          ) {
    for(int token_idx = blockIdx.x * blockDim.x + threadIdx.x; token_idx < token_num; token_idx += blockDim.x * gridDim.x) {
        const int ori_token_idx = token_idx + padding_offsets[token_idx];
        const int ori_bi = ori_token_idx / seq_len;
        const int local_token_id = ori_token_idx % seq_len;
        const int seq_len_decoder = seq_lens_decoder[ori_bi];


        int decoder_id  = -1;
        for (int i = 0; i <= ori_bi; i++) {
            decoder_id += (seq_len_decoder != 0);
        }
        int encoder_id = ori_bi - decoder_id - 1;

        if (seq_len_decoder > 0) {
            if (local_token_id == 0)
                insert_index_decoder[decoder_id] = ori_bi;
            logits_decoder_index[decoder_id * medusa_len + local_token_id] = token_idx;
        } else {
            if (local_token_id == 0)
                insert_index_encoder[encoder_id] = ori_bi;
            if (seq_lens_encoder[ori_bi] - 1 == local_token_id)
                logits_encoder_index[encoder_id] = token_idx;
        }
    }
}

std::vector<paddle::Tensor> MedusaRebuildLogits(const paddle::Tensor& seq_lens_encoder,
                                                            const paddle::Tensor& seq_lens_decoder,
                                                            const paddle::Tensor& padding_offsets,
                                                            const paddle::Tensor& input_ids,
                                                            const paddle::Tensor& retrieve_indices
                                                            ) {
    int token_num = padding_offsets.shape()[0];
    int medusa_len = retrieve_indices.shape()[1]-1;
    int max_seq_len = input_ids.shape()[1];;
    int bsz = input_ids.shape()[0];

    auto cu_stream = seq_lens_encoder.stream();

    int num_decoders = 0;
    int* d_num_decoders;
    cudaMalloc(&d_num_decoders, sizeof(int));
    GetNumDecoder<<<1, 1024, 0, cu_stream>>>(d_num_decoders, seq_lens_decoder.data<int>(), bsz);
    cudaMemcpy(&num_decoders, d_num_decoders, sizeof(int), cudaMemcpyDeviceToHost);

    int num_encoders = bsz - num_decoders;

    auto insert_index_encoder = paddle::empty({num_encoders}, paddle::DataType::INT32);
    auto insert_index_decoder = paddle::empty({num_decoders}, paddle::DataType::INT32);

    auto logits_encoder_index = paddle::empty({num_encoders}, paddle::DataType::INT32);
    auto logits_decoder_index = paddle::empty({num_decoders, medusa_len}, paddle::DataType::INT32);

    MedusaRebuildLogitsKernel<<<token_num, 512, 0, cu_stream>>>(logits_decoder_index.data<int>(),
                                                                logits_decoder_index.data<int>(),
                                                                insert_index_decoder.data<int>(),
                                                                insert_index_encoder.data<int>(),
                                                                seq_lens_encoder.data<int>(),
                                                                seq_lens_decoder.data<int>(),
                                                                padding_offsets.data<int>(),
                                                                token_num,
                                                                max_seq_len,
                                                                medusa_len
                                                                );

}




std::vector<std::vector<int64_t>> MedusaRebuildLogitInferShape(
                                                            const std::vector<int64_t>& seq_lens_encoder_shape,
                                                            const std::vector<int64_t>& seq_lens_decoder_shape,
                                                            const std::vector<int64_t>& padding_offsets_shape,
                                                            const std::vector<int64_t>& input_ids_shape,
                                                            const std::vector<int64_t>& retrieve_indices_shape) {
    int medusa_len = retrieve_indices_shape[1]-1;
    return {{-1, medusa_len}, {-1}, {-1}, {-1}};
}

std::vector<paddle::DataType> MedusaRebuildLogitsInferDtype(
                                                        const paddle::DataType& seq_lens_encoder_dtype,
                                                            const paddle::DataType& seq_lens_decoder_dtype,
                                                            const paddle::DataType& padding_offsets_dtype,
                                                            const paddle::DataType& input_ids_dtype,
                                                            const paddle::DataType& retrieve_indices_dtype) {
    return {paddle::DataType::INT32, paddle::DataType::INT32, paddle::DataType::INT32, paddle::DataType::INT32};
}


PD_BUILD_OP(medusa_rebuild_logits)
    .Inputs({"seq_lens_encoder", "seq_lens_decoder", "padding_offset", "input_ids", " retrieve_indices"})
    .Outputs({"logits_decoder_index",  "logits_encoder_index", "insert_index_decoder", "insert_index_encoder"})
    .SetKernelFn(PD_KERNEL(MedusaRebuildLogits))
    .SetInferShapeFn(PD_INFER_SHAPE(MedusaRebuildLogitInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MedusaRebuildLogitsInferDtype));