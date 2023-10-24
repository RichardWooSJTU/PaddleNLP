#include "helper.h"

template <int THREADBLOCK_SIZE>
__global__ void update_inputs_kernel(
    bool *not_need_stop,
    int *seq_lens_this_time,
    int *seq_lens_encoder,
    int *seq_lens_decoder,
    int64_t *input_ids,
    const int64_t *stop_nums,
    const bool *stop_flags,
    const int64_t *tree_candidates,
    const int bsz,
    const int input_ids_stride,
    const int medusa_len) {
  int thread_idx = threadIdx.x;
  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  bool stop_flag_now = false;
  int64_t stop_flag_now_int = 0;
    if (thread_idx < bsz) {
        stop_flag_now = stop_flags[thread_idx];
        stop_flag_now_int = static_cast<int64_t>(stop_flag_now);
    } else {
        return;
    }
  
  if (thread_idx < bsz) {
    const int seq_len_this_time = seq_lens_this_time[thread_idx];
    const int seq_len_encoder = seq_lens_encoder[thread_idx];
    const int seq_len_decoder = seq_lens_decoder[thread_idx];

    seq_lens_decoder[thread_idx] = stop_flag_now ? 0 : (seq_len_decoder == 0 ? seq_len_encoder : seq_len_decoder + 1);

    seq_lens_this_time[thread_idx] = stop_flag_now ? 0 : medusa_len;
    seq_lens_encoder[thread_idx] = 0;
    int64_t *input_ids_now = input_ids + thread_idx * input_ids_stride;
    for (int i = 0; i < medusa_len; i++) {
        input_ids_now[i] = tree_candidates[thread_idx * medusa_len + i];
    }
  }
  __syncthreads();
  int64_t stop_sum = BlockReduce(temp_storage).Sum(stop_flag_now_int);
  if (thread_idx == 0) {
    not_need_stop[0] = stop_sum < stop_nums[0];
  }
}

void MedusaUpdateInputes(const paddle::Tensor& stop_flags,
                   const paddle::Tensor& not_need_stop, // cpu
                   const paddle::Tensor& seq_lens_this_time,
                   const paddle::Tensor& seq_lens_encoder,
                   const paddle::Tensor& seq_lens_decoder,
                   const paddle::Tensor& input_ids,
                   const paddle::Tensor& stop_nums,
                   const paddle::Tensor& tree_candidates) {
  const int bsz = stop_flags.shape()[0];
  const int input_ids_stride = input_ids.shape()[1];
  const int medusa_len = tree_candidates.shape()[1];
  auto not_need_stop_gpu = not_need_stop.copy_to(stop_flags.place(), false);
  update_inputs_kernel<1024><<<1, 1024, 0, input_ids.stream()>>>(
    const_cast<bool*>(not_need_stop_gpu.data<bool>()),
    const_cast<int*>(seq_lens_this_time.data<int>()),
    const_cast<int*>(seq_lens_encoder.data<int>()),
    const_cast<int*>(seq_lens_decoder.data<int>()),
    const_cast<int64_t*>(input_ids.data<int64_t>()),
    stop_nums.data<int64_t>(),
    stop_flags.data<bool>(),
    tree_candidates.data<int64_t>(),
    bsz,
    input_ids_stride,
    medusa_len
  );
  auto not_need_stop_cpu = not_need_stop_gpu.copy_to(not_need_stop.place(), false);
  bool *not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
  not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}

PD_BUILD_OP(medusa_update_inputs)
    .Inputs({"stop_flags", 
             "not_need_stop", 
             "seq_lens_this_time", 
             "seq_lens_encoder", 
             "seq_lens_decoder",
             "input_ids",
             "stop_nums",
             "tree_candidates"})
    .Outputs({"not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "tree_candidates_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"tree_candidates", "tree_candidates_out"}})
    .SetKernelFn(PD_KERNEL(MedusaUpdateInputes));
