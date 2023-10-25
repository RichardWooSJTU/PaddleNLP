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
__global__ void MedusaUpdateMaskKernel(T* src_mask,
                                       const int* seq_lens_encoder,
                                       const int* seq_lens_decoder,
                                       const T* medusa_mask,
                                       T mask_value,
                                       T minus_one,
                                       int max_seq_len,
                                       int medusa_len) {
    int b_id = blockIdx.y;
    int q_id = blockIdx.x;

    int base_id = b_id * max_seq_len * max_seq_len + q_id * max_seq_len;
    int medusa_base_id = q_id * medusa_len;

    const int seq_len_encoder = seq_lens_encoder[b_id];
    const int seq_len_decoder = seq_lens_decoder[b_id];


    using Vec = AlignedVector<T, VecSize>;

    Vec out_vec;

    for (int idx = threadIdx.x * VecSize; idx < max_seq_len; idx += blockDim.x * VecSize) {
        if (seq_len_encoder > 0) {
#pragma unroll 
            for (int i = 0; i < VecSize; i++) {
                if (q_id < seq_len_encoder && idx + i < q_id)
                    out_vec[i] = 0;
                else 
                    out_vec[i] = mask_value;
            }
            Store<T, VecSize>(out_vec, src_mask + base_id + idx);
        } else if (seq_len_decoder > 0) {
            if (q_id < medusa_len) {
                Vec in_vec;
                Load<T, VecSize>(medusa_mask + idx - seq_len_decoder, &in_vec);
#pragma unroll 
                for (int i = 0; i < VecSize; i++) {
                    if (idx + i < seq_len_decoder) {
                        out_vec[i] = 0;
                    } else if (idx + i < seq_len_decoder + medusa_len) {
                        out_vec[i] = minus_one * (in_vec[i] + minus_one) * mask_value;
                    } else {
                        out_vec[i] = mask_value;
                    }
                }
            } else {
#pragma unroll 
                for (int i = 0; i < VecSize; i++) {
                    out_vec[i] = mask_value;
                }
            }
            Store<T, VecSize>(out_vec, src_mask + base_id + idx);
        }
    }
}

template <paddle::DataType D>
void LaunchMedusaUpdateMask(const paddle::Tensor& src_mask,  // [bsz, 1, max_seq_len, max_seq_len]
                      const paddle::Tensor& seq_lens_encoder,
                      const paddle::Tensor& seq_lens_decoder, 
                      const paddle::Tensor& medusa_mask,  // [bsz, 1, medusa_len, medusa_len]
                      float mask_value) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;  

    auto bsz = src_mask.shape()[0];
    auto max_seq_len = src_mask.shape()[2];
    auto medusa_len = medusa_mask.shape()[2];

    constexpr int PackSize = 16 / sizeof(data_t);
    auto cu_stream = src_mask.stream(); 
    dim3 grid(max_seq_len, bsz, 1);
    constexpr int BlockSize = 128;

    DataType_ mask_value_data = static_cast<DataType_>(mask_value);
    DataType_ minus_one_data = static_cast<DataType_>(-1.0f);

    MedusaUpdateMaskKernel<DataType_, PackSize><<<grid, BlockSize, 0, cu_stream>>>(
           reinterpret_cast<DataType_*>(const_cast<data_t*>(src_mask.data<data_t>())),
           seq_lens_encoder.data<int>(),
           seq_lens_decoder.data<int>(),
           reinterpret_cast<const DataType_*>(medusa_mask.data<data_t>()),  
           mask_value_data,
           minus_one_data,
           max_seq_len,
           medusa_len
    );
}

void MedusaUpdateMask(const paddle::Tensor& src_mask,  // [bsz, 1, max_seq_len, max_seq_len]
                      const paddle::Tensor& seq_lens_encoder,
                      const paddle::Tensor& seq_lens_decoder, 
                      const paddle::Tensor& medusa_mask,  // [bsz, 1, medusa_len, medusa_len]
                      float mask_value) {
    switch (src_mask.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchMedusaUpdateMask<paddle::DataType::BFLOAT16>(
                src_mask, seq_lens_encoder, seq_lens_decoder, medusa_mask, mask_value
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchMedusaUpdateMask<paddle::DataType::FLOAT16>(
                src_mask, seq_lens_encoder, seq_lens_decoder, medusa_mask, mask_value
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchMedusaUpdateMask<paddle::DataType::FLOAT32>(
                src_mask, seq_lens_encoder, seq_lens_decoder, medusa_mask, mask_value
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



PD_BUILD_OP(medusa_update_mask)
    .Inputs({"src_mask", 
             "seq_lens_encoder", 
             "seq_lens_decoder", 
             "medusa_mask"})
    .Outputs({"src_mask_out"})
    .Attrs({"mask_value: float"})
    .SetInplaceMap({{"src_mask", "src_mask_out"}})
    .SetKernelFn(PD_KERNEL(MedusaUpdateMask));
