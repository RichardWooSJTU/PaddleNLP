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

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *medusa_position_ids,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const int medusa_len) {
  using LoadT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 2 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    // if (seq_lens && seq_lens[ori_bi] == 0) continue;

    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;
    int emb_idx = 0;
    if (seq_lens_encoder[ori_bi] == 0) {
        // 得判断一下这里需要不需要减1
        emb_idx = seq_lens_decoder[ori_bi] + medusa_position_ids[ori_bi * medusa_len + ori_seq_id];
    } else {
        emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    }

    // const int64_t base_idx = token_idx * 3 * hidden_size + qkv_id *
    // hidden_size + hi * last_dim + h_bias;
    Load<T, VecSize>(&qkv[linear_index], &src_vec);
    Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = static_cast<float>(src_vec[2 * i]);
      const float input_right = static_cast<float>(src_vec[2 * i + 1]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      src_vec[2 * i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      src_vec[2 * i + 1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    Store<T, VecSize>(src_vec, &qkv_out[linear_index]);
  }
}


template <paddle::DataType D>
void LaunchMedusaRotaryQKVariable(const paddle::Tensor& qkv, 
              const paddle::Tensor& rotary_emb, 
              const paddle::Tensor& seq_lens_encoder,
              const paddle::Tensor& seq_lens_decoder,
              const paddle::Tensor& padding_offsets,
              const paddle::Tensor& medusa_position_ids,
              int num_head,
              bool use_neox_rotary_style) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;    

    const int token_num  = qkv.shape()[0];
    const int batch_size = rotary_emb.shape()[0];
    const int max_seq_len = rotary_emb.shape()[2];
    const int dim_head = rotary_emb.shape()[4];
    const int medusa_len = medusa_position_ids.shape()[1];

    int elem_nums = token_num * 3 * num_head * dim_head;  // just q and k
    if (use_neox_rotary_style) {
        elem_nums = token_num * 3 * num_head * dim_head / 2;
    }
    constexpr int PackSize = 16 / sizeof(data_t);
    const int pack_num = elem_nums / PackSize;
    const int blocksize = 128;
    int grid_size = 1;
    GetNumBlocks(pack_num, &grid_size);
    auto cu_stream = qkv.stream();

    if (!use_neox_rotary_style) {
        const float *cos_emb = rotary_emb.data<float>();
        const float *sin_emb = rotary_emb.data<float>() + max_seq_len * dim_head / 2;
        VariableLengthRotaryKernel<DataType_, PackSize>
            <<<grid_size, blocksize, 0, cu_stream>>>(reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
                                                            cos_emb,
                                                            sin_emb,
                                                            padding_offsets.data<int>(),
                                                            seq_lens_encoder.data<int>(),
                                                            seq_lens_decoder.data<int>(),
                                                            medusa_position_ids.data<int>(),
                                                            reinterpret_cast<DataType_*>(const_cast<data_t*>(qkv.data<data_t>())),
                                                            elem_nums,
                                                            num_head,
                                                            max_seq_len,
                                                            dim_head,
                                                            medusa_len);
    } else {
        // skip
    }
}

void MedusaRotaryQKVariable(const paddle::Tensor& qkv, 
              const paddle::Tensor& rotary_emb, 
              const paddle::Tensor& seq_lens_encoder,
              const paddle::Tensor& seq_lens_decoder,
              const paddle::Tensor& padding_offsets,
              const paddle::Tensor& medusa_position_ids,
              int num_head,
              bool use_neox_rotary_style) {
    switch (qkv.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchMedusaRotaryQKVariable<paddle::DataType::BFLOAT16>(
                qkv, rotary_emb, seq_lens_encoder, seq_lens_decoder, padding_offsets, medusa_position_ids, num_head, use_neox_rotary_style
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchMedusaRotaryQKVariable<paddle::DataType::FLOAT16>(
                qkv, rotary_emb, seq_lens_encoder, seq_lens_decoder, padding_offsets, medusa_position_ids, num_head, use_neox_rotary_style
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchMedusaRotaryQKVariable<paddle::DataType::FLOAT32>(
                qkv, rotary_emb, seq_lens_encoder, seq_lens_decoder, padding_offsets, medusa_position_ids, num_head,use_neox_rotary_style
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



PD_BUILD_OP(medusa_encode_rotary_qk_variable)
    .Inputs({"qkv", "rotary_emb", "seq_lens_encoder", "seq_lens_decoder", "padding_offsets", "medusa_position_ids"})
    .Outputs({"qkv_out"})
    .SetInplaceMap({{"qkv", "qkv_out"}})
    .Attrs({"num_head: int", "use_neox_rotary_style: bool"})
    .SetKernelFn(PD_KERNEL(MedusaRotaryQKVariable)); 