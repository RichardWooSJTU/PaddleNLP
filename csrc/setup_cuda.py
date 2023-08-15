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

from paddle.utils.cpp_extension import CUDAExtension, setup
import paddle

def get_gencode_flags():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]

gencode_flags = get_gencode_flags()

setup(
    name="paddlenlp_ops",
    ext_modules=CUDAExtension(
        sources=[
            "save_with_output.cc",
            "set_mask_value.cu",
            "set_value_by_flags.cu",
            "token_penalty_multi_scores.cu",
            "stop_generation_multi_ends.cu",
            "fused_get_rope.cu",
            "get_padding_offset.cu",
            "qkv_transpose_split.cu",
            "rebuild_padding.cu",
            "transpose_removing_padding.cu",
            "write_cache_kv.cu",
            "encode_rotary_qk.cu",
            "top_p_sampling.cu",
            "dequant_int8.cu",
            "quant_int8.cu",
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
        }
    ),
)
