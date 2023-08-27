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
from __future__ import annotations

import numpy as np
import json
import os
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddlenlp_ops import fused_get_rotary_embedding, get_padding_offset

from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedMultiTransformer,
    FusedMultiTransformerInt8
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationInferenceModel,
)
from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM, LlamaPretrainedModel
from paddlenlp.transformers.llama.modeling import LlamaLMHead
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import register_base_model

__all__ = ["LlamaInferenceModel", "LlamaForCausalLMInferenceModel"]

class load_act_scale_json:
    def __init__(
        self,
        scale_json_file_path="act_scales.json",
        key_map_dict=None,
        num_of_layers=None,
    ):
        with open(scale_json_file_path) as json_file:
            self.scale_dict = json.load(json_file)
        self.key_map = key_map_dict
        self.scale = {}
        for scale_type, key_template in self.key_map.items():
            self.scale[scale_type] = np.full([num_of_layers], fill_value=-1.0)
            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    self.scale[scale_type][i] = (
                        1 / self.scale_dict[key_template.replace("#", str(i))]
                    )


class load_weight_scale_json:
    def __init__(
        self,
        scale_json_file_path="weight_scales.json",
        key_map_dict=None,
        num_of_layers=None,
    ):
        with open(scale_json_file_path) as json_file:
            self.scale_dict = json.load(json_file)
        self.key_map = key_map_dict
        self.scale = {}
        for scale_type, key_template in self.key_map.items():
            # import pdb;pdb.set_trace()
            no_skip_layer_list = []
            n = 1
            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    no_skip_layer_list.append(key_template.replace("#", str(i)))
            if len(no_skip_layer_list) > 0:
                n = len(self.scale_dict[no_skip_layer_list[0]])
            print(f"scale_type:{scale_type}, cols:{n}")
            self.scale[scale_type] = np.full([num_of_layers, n], fill_value=-1.0, dtype='float32')
            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    self.scale[scale_type][i, :] = self.scale_dict[
                        key_template.replace("#", str(i))
                    ]
        # concat qkv and ffn1
        self.scale["qkv_weight_scale"] = []
        self.scale["ffn1_weight_scale"] = []
        for i in range(num_of_layers):
            print("concat ", i)
            self.scale["qkv_weight_scale"].append(np.concatenate([self.scale["q_weight_scale"][i, :],
                                                                 self.scale["k_weight_scale"][i, :],
                                                                 self.scale["v_weight_scale"][i, :]]))
            self.scale["ffn1_weight_scale"].append(np.concatenate([self.scale["ffn1_1_weight_scale"][i, :],
                                                                 self.scale["ffn1_2_weight_scale"][i, :]]))

class FusedLlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        result = paddle.incubate.nn.functional.fused_rms_norm(
            hidden_states, self.weight, None, self.variance_epsilon, begin_norm_axis=1
        )
        if isinstance(result, tuple):
            return result[0]
        return result


@register_base_model
class LlamaInferenceModel(LlamaPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.quant_model_path = config.quant_model_path
        self.quant_type = config.quant_type
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.epsilon = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings

        if config.tensor_parallel_degree > 1:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        # get ring_id
        ring_id = -1
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
        except:
            pass

        ln_scale_attrs = [paddle.ParamAttr(name="fusellama.{}.ln_scale".format(i)) for i in range(self.num_layers)]
        qkv_weight_attrs = [paddle.ParamAttr(name="fusellama.{}.qkv_weight".format(i)) for i in range(self.num_layers)]
        out_proj_weight_attrs = [
            paddle.ParamAttr(name="fusellama.{}.out_proj_weight".format(i)) for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn1_weight".format(i)) for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn2_weight".format(i)) for i in range(self.num_layers)
        ]

        if self.quant_type == "A8W8":
            qkv_weight_out_scale_attrs = [paddle.ParamAttr(name="fusellama.{}.qkv_weight_out_scale".format(i)) for i in range(self.num_layers)]
            linear_weight_out_scale_attrs = [paddle.ParamAttr(name="fusellama.{}.linear_weight_out_scale".format(i)) for i in range(self.num_layers)]
            ffn1_weight_out_scale_attrs = [paddle.ParamAttr(name="fusellama.{}.ffn1_weight_out_scale".format(i)) for i in range(self.num_layers)]
            ffn2_weight_out_scale_attrs = [paddle.ParamAttr(name="fusellama.{}.ffn2_weight_out_scale".format(i)) for i in range(self.num_layers)]

            linear_shift_attrs = [paddle.ParamAttr(name="fusellama.{}.linear_shift".format(i)) for i in range(self.num_layers)]
            linear_smooth_attrs = [paddle.ParamAttr(name="fusellama.{}.linear_smooth".format(i)) for i in range(self.num_layers)]
            ffn2_shift_attrs = [paddle.ParamAttr(name="fusellama.{}.ffn2_shift".format(i)) for i in range(self.num_layers)]
            ffn2_smooth_attrs = [paddle.ParamAttr(name="fusellama.{}.ffn2_smooth".format(i)) for i in range(self.num_layers)]


            self.transformer_block = FusedMultiTransformerInt8(
                self.hidden_size,
                self.num_attention_heads,
                self.intermediate_size,
                activation="swiglu",
                num_layers=config.num_hidden_layers,
                nranks=config.tensor_parallel_degree,
                ring_id=ring_id,
                ln_scale_attrs=ln_scale_attrs,
                qkv_weight_attrs=qkv_weight_attrs,
                linear_weight_attrs=out_proj_weight_attrs,
                ffn_ln_scale_attrs=ffn_ln_scale_attrs,
                ffn1_weight_attrs=ffn1_weight_attrs,
                ffn2_weight_attrs=ffn2_weight_attrs,
                qkv_weight_out_scale_attrs=qkv_weight_out_scale_attrs,
                linear_weight_out_scale_attrs=linear_weight_out_scale_attrs,
                ffn1_weight_out_scale_attrs=ffn1_weight_out_scale_attrs,
                ffn2_weight_out_scale_attrs=ffn2_weight_out_scale_attrs,
                linear_shift_attrs=linear_shift_attrs,
                linear_smooth_attrs=linear_smooth_attrs,
                ffn2_shift_attrs=ffn2_shift_attrs,
                ffn2_smooth_attrs=ffn2_smooth_attrs,
                epsilon=self.epsilon,
                norm_type="rmsnorm",
                use_neox_rotary_style=True,
            )
        else:
            self.transformer_block = FusedMultiTransformer(
                self.hidden_size,
                self.num_attention_heads,
                self.intermediate_size,
                activation="swiglu",
                num_layers=config.num_hidden_layers,
                nranks=config.tensor_parallel_degree,
                ring_id=ring_id,
                ln_scale_attrs=ln_scale_attrs,
                qkv_weight_attrs=qkv_weight_attrs,
                linear_weight_attrs=out_proj_weight_attrs,
                ffn_ln_scale_attrs=ffn_ln_scale_attrs,
                ffn1_weight_attrs=ffn1_weight_attrs,
                ffn2_weight_attrs=ffn2_weight_attrs,
                epsilon=self.epsilon,
                norm_type="rmsnorm",
                use_neox_rotary_style=True,
            )

        self.norm = FusedLlamaRMSNorm(config)

        self.cache_kvs = None
        self.head_dim_shape_tensor = paddle.ones((self.hidden_size // self.num_attention_heads), dtype="int8")

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        cache_kvs=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        past_key_values = kwargs.get("cache", None)
        is_decoder = past_key_values is not None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        cache_kvs = cache_kvs if cache_kvs is not None else self.cache_kvs

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len_encoder)
        else:
            ids_remove_padding = input_ids.squeeze(axis=1)
            padding_offset = None
            cum_offsets = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(ids_remove_padding)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder

        new_rope = fused_get_rotary_embedding(input_ids, position_ids, self.head_dim_shape_tensor, 0, True)
        with paddle.fluid.framework._stride_in_no_check_dy2st_diff():
            hidden_states, _ = self.transformer_block(
                input_ids,
                hidden_states,
                cum_offsets=cum_offsets,
                padding_offset=padding_offset,
                attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                caches=cache_kvs,
                seq_lens=seq_lens,
                rotary_embs=new_rope,
                rotary_emb_dims=1,
                time_step=paddle.increment(paddle.shape(attention_mask)[-1], -1) if is_decoder else None,
            )
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        unfused_state_dict = {}
        head_size = self.hidden_size // self.num_attention_heads

        self.embed_tokens.weight.set_value(paddle.to_tensor(state_dict["llama.embed_tokens.weight"]))
        self.norm.weight.set_value(paddle.to_tensor(state_dict["llama.norm.weight"]))

        for idx in range(self.config.num_hidden_layers):
            unfused_state_dict = {}
            unfused_state_dict["self_attn.q_proj.weight"] = state_dict[
                "llama.layers.{}.self_attn.q_proj.weight".format(idx)
            ]
            unfused_state_dict["self_attn.k_proj.weight"] = state_dict[
                "llama.layers.{}.self_attn.k_proj.weight".format(idx)
            ]
            unfused_state_dict["self_attn.v_proj.weight"] = state_dict[
                "llama.layers.{}.self_attn.v_proj.weight".format(idx)
            ]

            concated_qkv_weight = (
                np.concatenate(
                    [
                        unfused_state_dict["self_attn.q_proj.weight"],
                        unfused_state_dict["self_attn.k_proj.weight"],
                        unfused_state_dict["self_attn.v_proj.weight"],
                    ],
                    axis=-1,
                )
                .transpose(1, 0)
                .reshape(
                    3 * (self.num_attention_heads // self.config.tensor_parallel_degree) * (head_size),
                    self.hidden_size,
                )
            )  # reshape(3, self.num_attention_heself.hidden_sizeads // self.config.tensor_parallel_degree, head_size, )
            

            unfused_state_dict["mlp.gate_proj.weight"] = state_dict["llama.layers.{}.mlp.gate_proj.weight".format(idx)]
            unfused_state_dict["mlp.up_proj.weight"] = state_dict["llama.layers.{}.mlp.up_proj.weight".format(idx)]

            concated_ffn1_weight = np.concatenate(
                [unfused_state_dict["mlp.gate_proj.weight"], unfused_state_dict["mlp.up_proj.weight"]], axis=-1
            )

            if self.quant_type == "A8W8":
                self.transformer_block.qkv_weights[idx].set_value(
                    paddle.cast(paddle.to_tensor(concated_qkv_weight), 'int8')
                )
                self.transformer_block.linear_weights[idx].set_value(
                    paddle.cast(
                        paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)]).transpose((1, 0)),
                        'int8'
                    )
                )
                self.transformer_block.ffn1_weights[idx].set_value(
                    paddle.cast(
                        paddle.to_tensor(concated_ffn1_weight).transpose((1, 0)),
                        'int8'
                    )
                )
                self.transformer_block.ffn2_weights[idx].set_value(
                    paddle.cast(
                        paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)]).transpose((1, 0)),
                        'int8'
                    )
                )
                if "llama.layers.{}.self_attn.o_proj.shift_bias".format(idx) in state_dict:
                    self.transformer_block.linear_shifts[idx].set_value(
                        paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.shift_bias".format(idx)])
                    )
                    self.transformer_block.linear_smooths[idx].set_value(
                        paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.smooth_weight".format(idx)])
                    )
                    self.transformer_block.ffn2_shifts[idx].set_value(
                        paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.shift_bias".format(idx)])
                    )
                    self.transformer_block.ffn2_smooths[idx].set_value(
                        paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.smooth_weight".format(idx)])
                    )
                else:
                    self.transformer_block.linear_shifts[idx].set_value(
                        paddle.zeros(shape=[self.hidden_size]).astype(paddle.get_default_dtype())
                    )
                    self.transformer_block.linear_smooths[idx].set_value(
                        paddle.ones(shape=[self.hidden_size]).astype(paddle.get_default_dtype())
                    )
                    self.transformer_block.ffn2_shifts[idx].set_value(
                        paddle.zeros(shape=[self.intermediate_size]).astype(paddle.get_default_dtype())
                    )
                    self.transformer_block.ffn2_smooths[idx].set_value(
                        paddle.ones(shape=[self.intermediate_size]).astype(paddle.get_default_dtype())
                    )

            else:
                self.transformer_block.qkv_weights[idx].set_value(paddle.to_tensor(concated_qkv_weight))
                self.transformer_block.linear_weights[idx].set_value(
                    paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)])
                )
                self.transformer_block.ffn1_weights[idx].set_value(paddle.to_tensor(concated_ffn1_weight))
                self.transformer_block.ffn2_weights[idx].set_value(
                    paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)])
                )

            self.transformer_block.ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.input_layernorm.weight".format(idx)])
            )
            self.transformer_block.ffn_ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.post_attention_layernorm.weight".format(idx)])
            )
        if self.quant_type == "A8W8":
            with open('../paddlenlp/experimental/transformers/llama/ptq_scales_map.json') as json_file:
                scale_map_dict = json.load(json_file)
                act_scale_map_dict = scale_map_dict["act_scale"]
                weight_scale_map_dict = scale_map_dict["weight_scale"]
                #TODO(wufeisheng): support multi-cards

                act_scale_json_path = os.path.join(self.quant_model_path, f"act_scales.json")
                weight_scale_json_path = os.path.join(self.quant_model_path, f"weight_scales.json")
                act_scales = load_act_scale_json(
                    act_scale_json_path, act_scale_map_dict, num_of_layers=self.config.num_hidden_layers
                )
                self.transformer_block.act_scales = act_scales.scale

                weight_scales = load_weight_scale_json(
                    weight_scale_json_path, weight_scale_map_dict, num_of_layers=self.config.num_hidden_layers
                )
                for k,v in weight_scales.scale.items():
                    if 'qkv_' in k:
                        for i_layer, weight_scale in enumerate(v):
                            tmp = paddle.to_tensor(weight_scale/(127.0*127.0*act_scales.scale['qkv_in_scale'][i_layer])).reshape([self.num_attention_heads // self.config.tensor_parallel_degree,
                                        3,
                                        head_size,
                                    ]
                                ).transpose(
                                    [1, 0, 2]
                                ).reshape([-1])
                            self.transformer_block.qkv_weight_out_scales[i_layer].set_value(tmp)
                        pass
                    elif 'out_linear_' in k:
                        for i_layer, weight_scale in enumerate(v):
                            self.transformer_block.linear_weight_out_scales[i_layer].set_value(paddle.to_tensor(weight_scale/(127.0*127.0*act_scales.scale['out_linear_in_scale'][i_layer])))
                    elif 'ffn1_weight_scale' in k:
                        for i_layer, weight_scale in enumerate(v):
                            self.transformer_block.ffn1_weight_out_scales[i_layer].set_value(paddle.to_tensor(weight_scale/(127.0*127.0*act_scales.scale['ffn1_in_scale'][i_layer])))
                    elif 'ffn2' in k:
                        for i_layer, weight_scale in enumerate(v):
                            self.transformer_block.ffn2_weight_out_scales[i_layer].set_value(paddle.to_tensor(weight_scale/(127.0*127.0*act_scales.scale['ffn2_in_scale'][i_layer])))


class LlamaForCausalLMInferenceModel(GenerationInferenceModel, LlamaForCausalLM):
    """
    Dynamic Batching for LLaMA Model with pretraining tasks on top.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaInferenceModel(config)
        self.lm_head = LlamaLMHead(config)

    @classmethod
    def get_cache_kvs_shape(
        cls, config: LlamaConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for llama model

        Args:
            max_batch_size (int): the max batch size
            max_length (int | None, optional): the max_length of cache_kvs. Defaults to None.

        Returns:
            list[paddle.Tensor]: the list tensor shape for cache
        """
        if max_length is None:
            max_length = config.max_position_embeddings

        cache_kvs = []
        for _ in range(config.num_hidden_layers):
            cache_kvs.append(
                [
                    2,
                    max_batch_size,
                    config.num_attention_heads // max(config.tensor_parallel_degree, 1),
                    max_length,
                    config.hidden_size // config.num_attention_heads,
                ]
            )
        return cache_kvs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache_kvs,
        seq_len_encoder,
        seq_len_decoder,
        tgt_ids,
        tgt_pos,
        tgt_generation_mask,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        cache = kwargs.get("cache", None)
        if cache is not None:
            input_ids = tgt_ids
            position_ids = tgt_pos
            attention_mask = (tgt_generation_mask - 1) * 1e4
        else:
            attention_mask = (attention_mask - 1) * 1e4
        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_kvs": cache_kvs,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
            "cache": cache,
        }
        return model_inputs

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        cache=None,
        cache_kvs=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            cache_kvs=cache_kvs,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
        )

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = self.criterion(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.set_value(state_dict["lm_head.weight"])
        self.model.set_state_dict({k: state_dict[k] for k in state_dict.keys()})
