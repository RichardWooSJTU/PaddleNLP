# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def get_gencode_flags():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]


gencode_flags = get_gencode_flags()

setup(
    name="paddlenlp_ops",
    ext_modules=CUDAExtension(
        sources=[
            # "./generation/save_with_output.cc",
            # "./generation/set_mask_value.cu",
            # "./generation/set_value_by_flags.cu",
            # "./generation/token_penalty_multi_scores.cu",
            # "./generation/stop_generation_multi_ends.cu",
            # "./generation/fused_get_rope.cu",
            # "./generation/get_padding_offset.cu",
            # "./generation/qkv_transpose_split.cu",
            # "./generation/rebuild_padding.cu",
            # "./generation/transpose_removing_padding.cu",
            # "./generation/write_cache_kv.cu",
            # "./generation/encode_rotary_qk.cu",
            # "./generation/top_p_sampling.cu",
            # "./generation/set_alibi_mask_value.cu",
            # "./generation/get_padding_offset_v2.cu",
            # "./generation/rebuild_padding_v2.cu",
            # "./generation/set_value_by_flags_v2.cu",
            # "./generation/stop_generation_multi_ends_v2.cu",
            # "./generation/update_inputs.cu",
            # "./generation/get_output.cc",
            # "./generation/reset_need_stop_value.cc",
            # "./generation/save_with_output_msg.cc",
            # "./generation/write_int8_cache_kv.cu",
            # "./generation/medusa_rotary_qk_variable.cu",
            # "./generation/medusa_qkv_transpose_split_fetch_concat.cu",
            # "./generation/medusa_rebuild_logits.cu",
            # "./generation/medusa_set_stop_value_multi_ends.cu",
            # "./generation/medusa_save_with_output_msg.cc",
            # "./generation/medusa_write_kv.cu"
            # "./generation/medusa_gather.cu",
            "./generation/medusa_update_inputs.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ]
            + gencode_flags,
        },
    ),
)
