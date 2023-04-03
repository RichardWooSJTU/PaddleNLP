# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import paddle
import paddle.distributed.fleet as fleet
from collections.abc import Mapping, Sequence
from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer

class _StaticGuard(object):
    def __init__(self):
        pass

    def __enter__(self):
        paddle.enable_static()

    def __exit__(self, exc_type, exc_val, exc_tb):
        paddle.disable_static()


class InferenceEngine(object):
    """
    Model Parallel Inference Engine

    Args:
        model_dir (string): root directory of inference model
        mp_degree (int): model parallel size
        tensorrt_config (TensorRTConfig): configurations for TensorRT inference
    """

    def __init__(self, model_file, param_file, mp_degree=1, tensorrt_config=None):
        self.model_file = model_file
        self.param_file = param_file
        self.mp_degree = mp_degree
        self.tensorrt_config = tensorrt_config
        self.auto = False

        if mp_degree == 1:
            self.nranks = 1
            self.rank = 0
        else:
            self.nranks = fleet.worker_num()
            self.rank = fleet.worker_index()

        self._static_guard = _StaticGuard()
        with self._static_guard:
            self._init_predictor()

    def _init_predictor(self):
        config = paddle.inference.Config(self.model_file, self.param_file)

        config.enable_memory_optim()
        config.switch_ir_optim(True)
        if paddle.fluid.core.is_compiled_with_cuda():
            device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
            config.enable_use_gpu(100, device_id)
        elif paddle.fluid.core.is_compiled_with_xpu():
            device_id = int(os.environ.get("FLAGS_selected_xpus", 0))
            config.enable_xpu()
            config.set_xpu_device_id(device_id)

        # distributed config
        if self.mp_degree > 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.rank]

            dist_config = config.dist_config()
            dist_config.set_ranks(self.nranks, self.rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)

            if self.auto:
                config_fname = os.path.join(self.model_dir, "rank_mapping.csv")
            else:
                config_fname = self._generate_comm_init_config(self.rank, self.nranks)
            dist_config.set_comm_init_config(config_fname)
            config.set_dist_config(dist_config)

        # TensorRT config
        if self.tensorrt_config:
            config.enable_tensorrt_engine(
                max_batch_size=self.tensorrt_config.max_batch_size,
                workspace_size=self.tensorrt_config.workspace_size,
                min_subgraph_size=self.tensorrt_config.min_subgraph_size,
                precision_mode=self.tensorrt_config.precision,
                use_static=self.tensorrt_config.use_static,
                use_calib_mode=self.tensorrt_config.use_calib_mode,
            )

            if self.tensorrt_config.collect_shape:
                config.collect_shape_range_info(self.tensorrt_config.shape_range_info_filename)
            else:
                config.enable_tuned_tensorrt_dynamic_shape(self.tensorrt_config.shape_range_info_filename, True)

        self.predictor = paddle.inference.create_predictor(config)

    def input_names(self):
        return self.predictor.get_input_names()

    def output_names(self):
        return self.predictor.get_output_names()

    def predict(self, data):
        # data in dict/list format
        with self._static_guard:
            if isinstance(data, Sequence):
                if len(data) != len(self.input_names()):
                    raise ValueError()
                for d, name in zip(data, self.input_names()):
                    handle = self.predictor.get_input_handle(name)
                    handle.copy_from_cpu(np.array(d.copy()))
            elif isinstance(data, Mapping):
                # key check
                for k, v in data.items():
                    handle = self.predictor.get_input_handle(k)
                    handle.copy_from_cpu(np.array(v))
            else:
                raise ValueError()

            self.predictor.run()
            return {name: self.predictor.get_output_handle(name).copy_to_cpu() for name in self.output_names()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp_degree", default=1, type=int, help="")
    parser.add_argument(
        "--model_file", default="./inference/glm.pdmodel", type=str, help="model directory")
    parser.add_argument(
        "--param_file", default="./inference/glm.pdiparams", type=str, help="model directory")

    args = parser.parse_args()
    return args


def preprocess(tokenizer, input_text, src_length, tgt_length):
    input_text = [text.strip() + "[gMASK]" for text in input_text]
    inputs = tokenizer(
        input_text,
        return_tensors="np",
        add_special_tokens=True,
        padding=True,
        max_length=src_length,
        truncation=True,
        truncation_side="left",
    )
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=tgt_length)
    inputs_tensor = {}
    for key, value in inputs.items():
        inputs_tensor[key] = paddle.to_tensor(value)
    return inputs_tensor

def main():

    args = parse_args()

    fleet.init(is_collective=True)
    infer_engine = InferenceEngine(args.model_file, args.param_file)

    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b-chinese")
    input_text = ['答案：年基准利率4.35%，上下文：从实际看,贷款的基本条件是: 一是中国大陆居民,年龄在60岁以下; 二是有稳定的住址和工作或经营地点; 三是有稳定的收入来源; 四是无不良信用记录,贷款用途不能作为炒股,赌博等行为; 五是具有完全民事行为能力。在已知答案的前提下，问题：',
    "答案：U系列，上下文：U系列是最好的，采用国际顶尖技术（由格力自主研发）双级变频压缩机，提高压缩机运转效率，制冷制热能力更强劲；1赫兹变频技术，使空调相当于一个15 W电灯泡，更加节能省电；送风面积广，风力大；生态风，净化空气。非常不错，现在国美在做活动，可以了解一下。在已知答案的前提下，问题："
    ]
    inputs = preprocess(tokenizer=tokenizer, input_text=input_text, src_length=120, tgt_length=40)

    outs = infer_engine.predict(inputs)

    ids = list(outs.values())[0]
    for i in range(len(input_text)):
        
        out_ids = [int(x) for x in ids[i]]
        result = tokenizer.decode(out_ids)
        # result = input_text + result

        print('Prompt:', input_text[i])
        print('Generation:', result)


if __name__ == "__main__":
    main()
