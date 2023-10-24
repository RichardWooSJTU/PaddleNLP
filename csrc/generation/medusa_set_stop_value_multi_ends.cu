#include "paddle/extension.h"
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<stdio.h>

__device__ bool medusa_is_in_end(const int64_t id, const int64_t *end_ids, int length) {
    bool flag = false;
    for (int i = 0; i < length; i++) {
        if (id == end_ids[i]) {
            return true;
        }
    }
    return flag;
}

__global__ void MedusaSetFlagByValueKernel(
        bool *stop_flags,
        int64_t *next_tokens_ids,
        const int64_t *end_ids, 
        const int *seq_lens,
        const int *accept_lengths,
        const int* insert_index_decoder,
        const int num_decoders, 
        const int num_medusa,
        const int end_length) {
    int tid = threadIdx.x;
    if (tid < num_decoders) {
        const int accept_length = accept_lengths[tid];
        int bi = insert_index_decoder[tid];

        if (medusa_is_in_end(next_tokens_ids[tid * num_medusa + accept_length - 1], end_ids, end_length)) {
            stop_flags[bi] = true;
        }
    }
}

void MedusaGetStopFlagsMulti(const paddle::Tensor& next_tokens_ids, const paddle::Tensor& stop_flags, 
                             const paddle::Tensor& seq_lens, const paddle::Tensor& end_ids,  
                             const paddle::Tensor& accept_lengths,
                             const paddle::Tensor& insert_index_decoder) {
    PD_CHECK(next_tokens_ids.dtype() == paddle::DataType::INT64);
    PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
    
    auto cu_stream = next_tokens_ids.stream();
    std::vector<int64_t> shape = next_tokens_ids.shape();
    int64_t num_decoders = shape[0];
    int64_t num_medusa = shape[1];
    int64_t end_length = end_ids.shape()[0];
    int block_size = (num_decoders + 32 - 1) / 32 * 32;
    MedusaSetFlagByValueKernel<<<1, block_size, 0, cu_stream>>>(
        const_cast<bool*>(stop_flags.data<bool>()), 
        const_cast<int64_t*>(next_tokens_ids.data<int64_t>()),
        end_ids.data<int64_t>(),
        seq_lens.data<int>(),
        accept_lengths.data<int>(), 
        insert_index_decoder.data<int>(),
        num_decoders, num_medusa, end_length);
}

PD_BUILD_OP(medusa_set_stop_value_multi_ends)
    .Inputs({"next_tokens", "stop_flags", "seq_lens", "end_ids", "accept_lengths", "insert_index_decoder"})
    .Outputs({"next_tokens_ids_out", "stop_flags_out"})
    .SetInplaceMap({{"next_tokens", "next_tokens_ids_out"},
                    {"stop_flags", "stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(MedusaGetStopFlagsMulti));