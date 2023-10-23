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
from turtle import update

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.nn.quant import weight_quantize
from paddlenlp_ops import (
    fused_get_rotary_embedding,
    get_padding_offset,
    get_padding_offset_v2,
    medusa_rebuild_logits,
)

from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedBlockMultiTransformer,
    FusedMultiTransformer,
    FusedMedusaMultiTransformer
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationBlockInferenceModel,
    GenerationInferenceModel,
    GenerationMedusaBlockInferenceModel
)
from paddlenlp.transformers import LlamaConfig, LlamaPretrainedModel
from paddlenlp.transformers.llama.modeling import LlamaLMHead
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)

__all__ = [
    "LlamaInferenceModel",
    "LlamaBlockInferenceModel",
    "LlamaForCausalLMInferenceModel",
    "LlamaForCausalLMBlockInferenceModel",
    "LlamaForMiniGPT4InferenceModel",
]


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
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.epsilon = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        self.use_weight_only = False

        self.quant_bits = config.quant_bits
        self.quant_algo = "weight_only_int" + str(self.quant_bits)
        if self.quant_bits != -1:
            self.use_weight_only = True

        if self.use_weight_only:
            assert (
                self.quant_algo == "weight_only_int8" or self.quant_algo == "weight_only_int4"
            ), "Expected quant_algo equal to 'weight_only_int8' or 'weight_only_int4', but received {}".format(
                self.quant_algo
            )

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
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None

        if self.use_weight_only:
            qkv_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.qkv_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            out_proj_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.out_proj_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn1_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn2_weight_scale".format(i)) for i in range(self.num_layers)
            ]

        self.transformer_block = FusedMultiTransformer(
            self.hidden_size,
            self.num_attention_heads,
            self.intermediate_size,
            quant_bits=self.quant_bits,
            activation="swiglu",
            num_layers=config.num_hidden_layers,
            nranks=config.tensor_parallel_degree,
            ring_id=ring_id,
            ln_scale_attrs=ln_scale_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_weight_scale_attrs=qkv_weight_scale_attrs,
            linear_weight_attrs=out_proj_weight_attrs,
            linear_weight_scale_attrs=out_proj_weight_scale_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
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

    # This function is a little different from prepare_input_ids_for_generation in paddlenlp/transformers/generation/utils.py
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        cache_kvs=None,
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
        **kwargs,
    ):
        # kwargs["cache"] is used used to distinguish between encoder and decoder phase.
        past_key_values = kwargs.get("cache", None)
        is_decoder = past_key_values is not None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # genereate a fake input_ids according to inputs_embeds
        # this is usually occurred in img2txt multimodal model when first enter into this forward function.
        if input_ids is None and inputs_embeds is not None:
            input_ids = self.prepare_input_ids_for_generation(self.config.bos_token_id, inputs_embeds)
        if inputs_embeds is not None:
            batch, seq_len, hidden_dim = inputs_embeds.shape
            # merge batch and seq_len dimension.
            inputs_embeds = inputs_embeds.reshape([batch * seq_len, hidden_dim])

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
            ids_remove_padding = input_ids
            padding_offset = None
            cum_offsets = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(ids_remove_padding)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder

        position_offset = 0
        if not is_decoder and pre_caches is not None:
            position_offset = 128
        new_rope = fused_get_rotary_embedding(
            input_ids, position_ids, self.head_dim_shape_tensor, position_offset, True
        )

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids,
                hidden_states,
                cum_offsets=cum_offsets,
                padding_offset=padding_offset,
                attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                caches=cache_kvs,
                pre_caches=pre_caches,
                pre_caches_length=position_offset,
                seq_lens=seq_lens,
                rotary_embs=new_rope,
                rotary_emb_dims=1,
                time_step=paddle.increment(paddle.shape(attention_mask)[-1], -1) if is_decoder else None,
                k_quant_scales=k_quant_scales,
                v_quant_scales=v_quant_scales,
                k_dequant_scales=k_dequant_scales,
                v_dequant_scales=v_dequant_scales,
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

            qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
            if self.use_weight_only:
                qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
                qkv_weight_tensor = paddle.transpose(qkv_weight_tensor, perm=[1, 0])
                qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                    qkv_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.qkv_weights[idx].set_value(qkv_quanted_weight_tensor)
                self.transformer_block.qkv_weights_scale[idx].set_value(qkv_weight_scale_tensor)
            else:
                self.transformer_block.qkv_weights[idx].set_value(qkv_weight_tensor)

            linear_weight_tensor = paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)])
            if self.use_weight_only:
                linear_quanted_weight_tensor, linear_weight_scale_tensor = weight_quantize(
                    linear_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight_tensor)
                self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale_tensor)
            else:
                self.transformer_block.linear_weights[idx].set_value(linear_weight_tensor)

            unfused_state_dict["mlp.gate_proj.weight"] = state_dict["llama.layers.{}.mlp.gate_proj.weight".format(idx)]
            unfused_state_dict["mlp.up_proj.weight"] = state_dict["llama.layers.{}.mlp.up_proj.weight".format(idx)]

            concated_ffn1_weight = np.concatenate(
                [unfused_state_dict["mlp.gate_proj.weight"], unfused_state_dict["mlp.up_proj.weight"]], axis=-1
            )
            ffn1_weight_tensor = paddle.to_tensor(concated_ffn1_weight)

            if self.use_weight_only:
                ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = weight_quantize(
                    ffn1_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_quanted_weight_tensor)
                self.transformer_block.ffn1_weights_scale[idx].set_value(ffn1_weight_scale_tensor)
            else:
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_weight_tensor)

            ffn2_weight_tensor = paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)])
            if self.use_weight_only:
                ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = weight_quantize(
                    ffn2_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight_tensor)
                self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale_tensor)
            else:
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_weight_tensor)

            self.transformer_block.ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.input_layernorm.weight".format(idx)])
            )

            self.transformer_block.ffn_ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.post_attention_layernorm.weight".format(idx)])
            )


@register_base_model
class LlamaBlockInferenceModel(LlamaPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.epsilon = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        self.use_weight_only = False

        self.max_input_length = config.max_input_length
        self.block_size = config.block_size
        self.use_neox_rotary_style = True

        self.quant_bits = config.quant_bits
        self.quant_algo = "weight_only_int" + str(self.quant_bits)
        if self.quant_bits != -1:
            self.use_weight_only = True

        if self.use_weight_only:
            assert (
                self.quant_algo == "weight_only_int8" or self.quant_algo == "weight_only_int4"
            ), "Expected quant_algo equal to 'weight_only_int8' or 'weight_only_int4', but received {}".format(
                self.quant_algo
            )

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
        self.mp_rank = -1
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
            self.mp_rank = paddle.distributed.get_rank()
        except:
            pass

        ln_scale_attrs = [paddle.ParamAttr(name="fusellama.{}.ln_scale".format(i)) for i in range(self.num_layers)]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None

        if self.use_weight_only:
            qkv_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.qkv_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            out_proj_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.out_proj_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn1_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn2_weight_scale".format(i)) for i in range(self.num_layers)
            ]

        self.transformer_block = FusedBlockMultiTransformer(
            self.hidden_size,
            self.num_attention_heads,
            self.intermediate_size,
            quant_bits=self.quant_bits,
            activation="swiglu",
            num_layers=config.num_hidden_layers,
            nranks=config.tensor_parallel_degree,
            ring_id=ring_id,
            ln_scale_attrs=ln_scale_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_weight_scale_attrs=qkv_weight_scale_attrs,
            linear_weight_attrs=out_proj_weight_attrs,
            linear_weight_scale_attrs=out_proj_weight_scale_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            epsilon=self.epsilon,
            norm_type="rmsnorm",
            use_neox_rotary_style=self.use_neox_rotary_style,
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
        cum_offsets_now = paddle.cumsum(self.max_input_length - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        ids_remove_padding, cum_offsets, padding_offset, cu_seqlens_q, cu_seqlens_k = get_padding_offset_v2(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k

    # This function is a little different from prepare_input_ids_for_generation in paddlenlp/transformers/generation/utils.py
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    def forward(
        self,
        input_ids=None,
        src_mask=None,
        caches=None,
        rope_emb=None,
        block_tables=None,
        seq_lens_encoder=None,
        seq_lens_decoder=None,
        seq_lens_this_time=None,
        pre_key_caches=None,
        pre_value_caches=None,
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
    ):
        ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k = self.remove_padding(
            input_ids, seq_lens_this_time
        )

        inputs_embeds = self.embed_tokens(ids_remove_padding)

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                inputs_embeds,
                cum_offsets=cum_offsets,
                padding_offsets=padding_offset,
                attn_mask=src_mask,
                caches=caches,
                pre_key_caches=pre_key_caches,
                pre_value_caches=pre_value_caches,
                rotary_embs=rope_emb,
                seq_lens_encoder=seq_lens_encoder,
                seq_lens_decoder=seq_lens_decoder,
                seq_lens_this_time=seq_lens_this_time,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                k_quant_scales=k_quant_scales,
                v_quant_scales=v_quant_scales,
                k_dequant_scales=k_dequant_scales,
                v_dequant_scales=v_dequant_scales,
                block_tables=block_tables,
                max_input_length=self.max_input_length,
                block_size=self.block_size,
                use_neox_rotary_style=self.use_neox_rotary_style,
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
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

            qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
            if self.use_weight_only:
                qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
                qkv_weight_tensor = paddle.transpose(qkv_weight_tensor, perm=[1, 0])
                qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                    qkv_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.qkv_weights[idx].set_value(qkv_quanted_weight_tensor)
                self.transformer_block.qkv_weights_scale[idx].set_value(qkv_weight_scale_tensor)
            else:
                self.transformer_block.qkv_weights[idx].set_value(qkv_weight_tensor)

            linear_weight_tensor = paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)])
            if self.use_weight_only:
                linear_quanted_weight_tensor, linear_weight_scale_tensor = weight_quantize(
                    linear_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight_tensor)
                self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale_tensor)
            else:
                self.transformer_block.linear_weights[idx].set_value(linear_weight_tensor)

            unfused_state_dict["mlp.gate_proj.weight"] = state_dict["llama.layers.{}.mlp.gate_proj.weight".format(idx)]
            unfused_state_dict["mlp.up_proj.weight"] = state_dict["llama.layers.{}.mlp.up_proj.weight".format(idx)]

            concated_ffn1_weight = np.concatenate(
                [unfused_state_dict["mlp.gate_proj.weight"], unfused_state_dict["mlp.up_proj.weight"]], axis=-1
            )
            ffn1_weight_tensor = paddle.to_tensor(concated_ffn1_weight)

            if self.use_weight_only:
                ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = weight_quantize(
                    ffn1_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_quanted_weight_tensor)
                self.transformer_block.ffn1_weights_scale[idx].set_value(ffn1_weight_scale_tensor)
            else:
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_weight_tensor)

            ffn2_weight_tensor = paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)])
            if self.use_weight_only:
                ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = weight_quantize(
                    ffn2_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight_tensor)
                self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale_tensor)
            else:
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_weight_tensor)

            self.transformer_block.ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.input_layernorm.weight".format(idx)])
            )

            self.transformer_block.ffn_ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.post_attention_layernorm.weight".format(idx)])
            )


class LlamaForCausalLMInferenceModel(GenerationInferenceModel, LlamaPretrainedModel):
    """
    Dynamic Batching for LLaMA Model with pretraining tasks on top.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.llama = LlamaInferenceModel(config)
        self.lm_head = LlamaLMHead(config)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, from_hf_hub: bool = False, subfolder: str | None = None, *args, **kwargs
    ):
        # TODO: Support safetensors loading.
        kwargs["use_safetensors"] = False
        return super().from_pretrained(pretrained_model_name_or_path, from_hf_hub, subfolder, *args, **kwargs)

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
        pre_caches = kwargs.get("pre_caches", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)
        if cache is not None:
            input_ids = tgt_ids
            position_ids = tgt_pos
            attention_mask = (tgt_generation_mask - 1) * 1e4
            # make inputs_embeds be none in decoder phase.
            # in forward function, it will be assigned according to input_ids.
            inputs_embeds = None
        else:
            attention_mask = (attention_mask - 1) * 1e4
        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_kvs": cache_kvs,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
            "cache": cache,
            "pre_caches": pre_caches,
            "k_quant_scales": k_quant_scales,
            "v_quant_scales": v_quant_scales,
            "k_dequant_scales": k_dequant_scales,
            "v_dequant_scales": v_dequant_scales,
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
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.llama(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            cache_kvs=cache_kvs,
            pre_caches=pre_caches,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            k_quant_scales=k_quant_scales,
            v_quant_scales=v_quant_scales,
            k_dequant_scales=k_dequant_scales,
            v_dequant_scales=v_dequant_scales,
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
        self.llama.set_state_dict({k: state_dict[k] for k in state_dict.keys()})


class LlamaForCausalLMBlockInferenceModel(GenerationBlockInferenceModel, LlamaPretrainedModel):
    """
    Dynamic Batching for LLaMA Model with pretraining tasks on top.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.llama = LlamaBlockInferenceModel(config)
        self.lm_head = LlamaLMHead(config)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, from_hf_hub: bool = False, subfolder: str | None = None, *args, **kwargs
    ):
        # TODO: Support safetensors loading.
        kwargs["use_safetensors"] = False
        return super().from_pretrained(pretrained_model_name_or_path, from_hf_hub, subfolder, *args, **kwargs)

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
        max_block_per_seq = (config.max_seq_len + config.block_size - 1) // config.block_size
        if max_batch_size == -1:
            max_block_nums = None
        else:
            max_block_nums = max_batch_size * max_block_per_seq

        cache_kvs = []
        for _ in range(config.num_hidden_layers):
            cache_kv_shape = [
                max_block_nums,
                config.num_attention_heads // max(config.tensor_parallel_degree, 1),
                config.block_size,
                config.hidden_size // config.num_attention_heads,
            ]
            cache_kvs.append(cache_kv_shape)
            cache_kvs.append(cache_kv_shape)
        return cache_kvs

    def prepare_inputs_for_generation(self, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        input_ids = kwargs["input_ids"]
        src_mask = kwargs.get("src_mask", None)
        block_tables = kwargs.get("block_tables", None)

        pre_key_caches = kwargs.get("pre_key_caches", None)
        pre_value_caches = kwargs.get("pre_value_caches", None)
        caches = kwargs.get("caches", None)

        rope_emb = kwargs["rope_emb"]
        seq_lens_this_time = kwargs["seq_lens_this_time"]
        seq_lens_encoder = kwargs["seq_lens_encoder"]
        seq_lens_decoder = kwargs["seq_lens_decoder"]
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)
        model_inputs = {
            "input_ids": input_ids,
            "src_mask": src_mask,
            "rope_emb": rope_emb,
            "pre_key_caches": pre_key_caches,
            "pre_value_caches": pre_value_caches,
            "caches": caches,
            "seq_lens_this_time": seq_lens_this_time,
            "seq_lens_encoder": seq_lens_encoder,
            "seq_lens_decoder": seq_lens_decoder,
            "block_tables": block_tables,
            "k_quant_scales": k_quant_scales,
            "v_quant_scales": v_quant_scales,
            "k_dequant_scales": k_dequant_scales,
            "v_dequant_scales": v_dequant_scales,
        }
        return model_inputs

    def forward(
        self,
        input_ids,
        src_mask=None,
        pre_key_caches=None,
        pre_value_caches=None,
        caches=None,
        seq_lens_this_time=None,
        seq_lens_encoder=None,
        seq_lens_decoder=None,
        rope_emb=None,
        block_tables=None,
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
    ):
        outputs = self.llama(
            input_ids,
            src_mask=src_mask,
            caches=caches,
            rope_emb=rope_emb,
            block_tables=block_tables,
            pre_key_caches=pre_key_caches,
            pre_value_caches=pre_value_caches,
            seq_lens_this_time=seq_lens_this_time,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            k_quant_scales=k_quant_scales,
            v_quant_scales=v_quant_scales,
            k_dequant_scales=k_dequant_scales,
            v_dequant_scales=v_dequant_scales,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
        )

        return logits

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.set_value(state_dict["lm_head.weight"])
        self.llama.set_state_dict({k: state_dict[k] for k in state_dict.keys()})



class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        weight_attr = paddle.ParamAttr(
            name="weight",
            initializer=paddle.nn.initializer.Constant(value=0.0))
        bias_attr = paddle.ParamAttr(
            name="bias",
            initializer=paddle.nn.initializer.Constant(value=1.0))
        self.linear = paddle.nn.Linear(hidden_size, hidden_size, weight_attr, bias_attr)
        # Initialize as an identity mapping
        # Use SiLU activation to keep consistent with the Llama model
        self.act = paddle.nn.Silu()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


@register_base_model
class LlamaMedusaBlockInferenceModel(LlamaPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.epsilon = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        self.use_weight_only = False

        self.max_input_length = config.max_input_length
        self.block_size = config.block_size
        self.use_neox_rotary_style = True

        self.quant_bits = config.quant_bits
        self.quant_algo = "weight_only_int" + str(self.quant_bits)
        if self.quant_bits != -1:
            self.use_weight_only = True

        if self.use_weight_only:
            assert (
                self.quant_algo == "weight_only_int8" or self.quant_algo == "weight_only_int4"
            ), "Expected quant_algo equal to 'weight_only_int8' or 'weight_only_int4', but received {}".format(
                self.quant_algo
            )

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
        self.mp_rank = -1
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
            self.mp_rank = paddle.distributed.get_rank()
        except:
            pass

        ln_scale_attrs = [paddle.ParamAttr(name="fusellama.{}.ln_scale".format(i)) for i in range(self.num_layers)]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None

        if self.use_weight_only:
            qkv_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.qkv_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            out_proj_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.out_proj_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn1_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn2_weight_scale".format(i)) for i in range(self.num_layers)
            ]

        self.transformer_block = FusedMedusaMultiTransformer(
            self.hidden_size,
            self.num_attention_heads,
            self.intermediate_size,
            quant_bits=self.quant_bits,
            activation="swiglu",
            num_layers=config.num_hidden_layers,
            nranks=config.tensor_parallel_degree,
            ring_id=ring_id,
            ln_scale_attrs=ln_scale_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_weight_scale_attrs=qkv_weight_scale_attrs,
            linear_weight_attrs=out_proj_weight_attrs,
            linear_weight_scale_attrs=out_proj_weight_scale_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            epsilon=self.epsilon,
            norm_type="rmsnorm",
            use_neox_rotary_style=self.use_neox_rotary_style,
        )
        self.norm = FusedLlamaRMSNorm(config)

        self.cache_kvs = None
        self.head_dim_shape_tensor = paddle.ones((self.hidden_size // self.num_attention_heads), dtype="int8")

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

   

    # This function is a little different from prepare_input_ids_for_generation in paddlenlp/transformers/generation/utils.py
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    def forward(
        self,
        ids_remove_padding=None,
        cum_offsets=None,
        padding_offset=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        src_mask=None,
        medusa_position_ids=None,
        cache_k=None,
        cache_v=None,
        medusa_k=None,
        medusa_v=None,
        rope_emb=None,
        block_tables=None,
        seq_lens_encoder=None,
        seq_lens_decoder=None,
        seq_lens_this_time=None,
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
    ):
        # VERIFYING(@wufeisheng): modify this custom operators
       

        inputs_embeds = self.embed_tokens(ids_remove_padding)

        with dy2st_nocheck_guard_context():
            hidden_states = self.transformer_block(
                inputs_embeds,
                input_ids,
                cum_offsets=cum_offsets,
                padding_offsets=padding_offset,
                attn_mask=src_mask,
                medusa_position_ids=medusa_position_ids,
                cache_k=cache_k,
                cache_v=cache_v,
                medusa_k=medusa_k,
                medusa_v=medusa_v,
                rotary_embs=rope_emb,
                seq_lens_encoder=seq_lens_encoder,
                seq_lens_decoder=seq_lens_decoder,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                k_quant_scales=k_quant_scales,
                v_quant_scales=v_quant_scales,
                k_dequant_scales=k_dequant_scales,
                v_dequant_scales=v_dequant_scales,
                block_tables=block_tables,
                block_size=self.block_size,
                use_neox_rotary_style=self.use_neox_rotary_style,
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
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

            qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
            if self.use_weight_only:
                qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
                qkv_weight_tensor = paddle.transpose(qkv_weight_tensor, perm=[1, 0])
                qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                    qkv_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.qkv_weights[idx].set_value(qkv_quanted_weight_tensor)
                self.transformer_block.qkv_weights_scale[idx].set_value(qkv_weight_scale_tensor)
            else:
                self.transformer_block.qkv_weights[idx].set_value(qkv_weight_tensor)

            linear_weight_tensor = paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)])
            if self.use_weight_only:
                linear_quanted_weight_tensor, linear_weight_scale_tensor = weight_quantize(
                    linear_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight_tensor)
                self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale_tensor)
            else:
                self.transformer_block.linear_weights[idx].set_value(linear_weight_tensor)

            unfused_state_dict["mlp.gate_proj.weight"] = state_dict["llama.layers.{}.mlp.gate_proj.weight".format(idx)]
            unfused_state_dict["mlp.up_proj.weight"] = state_dict["llama.layers.{}.mlp.up_proj.weight".format(idx)]

            concated_ffn1_weight = np.concatenate(
                [unfused_state_dict["mlp.gate_proj.weight"], unfused_state_dict["mlp.up_proj.weight"]], axis=-1
            )
            ffn1_weight_tensor = paddle.to_tensor(concated_ffn1_weight)

            if self.use_weight_only:
                ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = weight_quantize(
                    ffn1_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_quanted_weight_tensor)
                self.transformer_block.ffn1_weights_scale[idx].set_value(ffn1_weight_scale_tensor)
            else:
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_weight_tensor)

            ffn2_weight_tensor = paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)])
            if self.use_weight_only:
                ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = weight_quantize(
                    ffn2_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight_tensor)
                self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale_tensor)
            else:
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_weight_tensor)

            self.transformer_block.ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.input_layernorm.weight".format(idx)])
            )

            self.transformer_block.ffn_ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.post_attention_layernorm.weight".format(idx)])
            )

class LlamaForCasualMedusaBlockInferenceModel(GenerationMedusaBlockInferenceModel, LlamaPretrainedModel):
    """
    Dynamic Batching for LLaMA Model with pretraining tasks on top.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.medusa = config.medusa_num_heads
        self.medusa_num_layers = config.medusa_num_layers

        self.llama = LlamaBlockInferenceModel(config)
        self.lm_head = LlamaLMHead(config)

        weight_attr = paddle.ParamAttr(
            name="weight",
            initializer=paddle.nn.initializer.Constant(value=0.0))
        

        self.medusa_head = paddle.nn.LayerList(
            [
                paddle.nn.Sequential(
                    *([ResBlock(self.hidden_size)] * config.medusa_num_layers),
                    paddle.nn.Linear(self.hidden_size, self.vocab_size, weight_attr=weight_attr, bias_attr=False),
                )
                for _ in range(config.medusa_num_heads)
            ]
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, from_hf_hub: bool = False, subfolder: str | None = None, *args, **kwargs
    ):
        # TODO: Support safetensors loading.
        kwargs["use_safetensors"] = False
        return super().from_pretrained(pretrained_model_name_or_path, from_hf_hub, subfolder, *args, **kwargs)

        # TODO(@wufeisheng): Load medusa Head weight

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
        max_block_per_seq = (config.max_seq_len + config.block_size - 1) // config.block_size
        if max_batch_size == -1:
            max_block_nums = None
        else:
            max_block_nums = max_batch_size * max_block_per_seq

        return [config.num_hidden_layers, max_block_nums, config.num_attention_heads // max(config.tensor_parallel_degree, 1), config.block_size,config.hidden_size // config.num_attention_heads]

    @classmethod
    def get_medusa_kvs_shape(
        cls, config: LlamaConfig, max_batch_size: int = None, medusa_len: int = None
    ) -> list[list[int]]:
        return [config.num_hidden_layers, max_batch_size, config.num_attention_heads // max(config.tensor_parallel_degree, 1), medusa_len, config.hidden_size // config.num_attention_heads]
        

    def prepare_inputs_for_generation(self, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        input_ids = kwargs["input_ids"]
        src_mask = kwargs.get("src_mask", None)
        block_tables = kwargs.get("block_tables", None)
        caches = kwargs.get("caches", None)
        medusa_k = kwargs.get("medusa_k", None) #输入：占位就可以了 后处理的时候更新到cachekv里面 
        medusa_v = kwargs.get("medusa_v", None)


        seq_lens_encoder = kwargs["seq_lens_encoder"]
        seq_lens_decoder = kwargs["seq_lens_decoder"]

        # TODO(@wufeisheng): 实现对src_mask的更新，如果是encoder 设置为casual，否则设为medusa模式
        # medusa模式需要根据seq_len来更新位置，但是[-medusa_len:]位置的mask为恒定值
        update_mask(src_mask, seq_lens_encoder, seq_lens_decoder, medusa_mask)

        rope_emb = kwargs["rope_emb"]
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)
        model_inputs = {
            "input_ids": input_ids,
            "src_mask": src_mask,
            "rope_emb": rope_emb,
            "cache_k": cache_k,
            "cache_v": cache_v,
            "medusa_k": medusa_k,
            "medusa_v": medusa_v,
            "seq_lens_encoder": seq_lens_encoder,
            "seq_lens_decoder": seq_lens_decoder,
            "block_tables": block_tables,
            "k_quant_scales": k_quant_scales,
            "v_quant_scales": v_quant_scales,
            "k_dequant_scales": k_dequant_scales,
            "v_dequant_scales": v_dequant_scales,
        }
        return model_inputs

    def remove_padding(self, input_ids, seq_lens_this_time, seq_lens_decoder):
        cum_offsets_now = paddle.cumsum(self.max_input_length - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        ids_remove_padding, cum_offsets, padding_offset, cu_seqlens_q, cu_seqlens_k = get_padding_offset_v2(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time, seq_lens_decoder
        )
        return ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k

    def forward(
        self,
        input_ids,
        src_mask=None,
        medusa_position_ids=None,
        pre_key_caches=None,
        pre_value_caches=None,
        cache_k=None,
        cache_v=None,
        medusa_k=None,
        medusa_v=None,
        seq_lens_this_time=None,
        seq_lens_encoder=None,
        seq_lens_decoder=None,
        rope_emb=None,
        block_tables=None,
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
    ):
        # Tree decoding: 
        ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k = self.remove_padding(
            input_ids, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder
        )

        outputs = self.llama(
            ids_remove_padding,
            padding_offset=padding_offset,
            cum_offsets=cum_offsets,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            src_mask=src_mask,
            medusa_position_ids=medusa_position_ids,
            cache_k=cache_k,
            cache_v=cache_v,
            medusa_k=medusa_k,
            medusa_v=medusa_v,
            rope_emb=rope_emb,
            block_tables=block_tables,
            seq_lens_this_time=seq_lens_this_time,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            k_quant_scales=k_quant_scales,
            v_quant_scales=v_quant_scales,
            k_dequant_scales=k_dequant_scales,
            v_dequant_scales=v_dequant_scales,
        )
        hidden_states = outputs[0] # [token_num, dim_embed] token_num = sum(seq_lens_this_time)

        medusa_logits= []
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))


        combined_medusa_logits = paddle.stack(medusa_logits, dim=0)# [4, token_num, vocab_size]

        orig_logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
        ) # [token_num, vocab_size]

        return orig_logits, combined_medusa_logits, padding_offset, cum_offsets
    
    def evaluate_posterior(
        self, logits, candidates, temperature, posterior_threshold, posterior_alpha
    ):
        """
        Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

        Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
        probabilities to select the best candidate.

        Args:
        - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
        - candidates (torch.Tensor): Candidate token sequences.
        - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
        - posterior_threshold (float): Threshold for posterior probability.
        - posterior_alpha (float): Scaling factor for the threshold.

        Returns:
        - best_candidate (torch.Tensor): Index of the chosen best candidate.
        - accept_length (int): Length of the accepted candidate sequence.
        """
        
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_prob = paddle.softmax(logits[:, :-1] / temperature, dim=-1)
        candidates_prob = paddle.gather(
            posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        posterior_entropy = -paddle.sum(
            posterior_prob * paddle.log(posterior_prob + 1e-5), dim=-1
        )  
        threshold = paddle.minimum(
            paddle.ones_like(posterior_entropy) * posterior_threshold,
            paddle.exp(-posterior_entropy) * posterior_alpha,
        )
        posterior_mask = candidates_prob > threshold
        candidates_accept_length = (paddle.cumprod(posterior_mask, dim=1)).sum(dim=1)

        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max()
        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate = paddle.to_tensor([0], dtype='int64', device=candidates.device)
        else:
            best_candidates = paddle.where(candidates_accept_length == accept_length)[0]
            # Accept the best one according to likelihood
            likelihood = paddle.sum(
                paddle.log(candidates_prob[best_candidates, :accept_length]), dim=-1
            )
            best_candidate = best_candidates[paddle.argmax(likelihood)]

        return best_candidate, accept_length

    def generate_candidates(self, medusa_logits, logits, tree_indices, retrieve_indices):
        """
        Generate candidates based on provided logits and indices.
        
        Parameters:
        - medusa_logits (torch.Tensor): Logits associated with the Medusa structure.
        - logits (torch.Tensor): Original logits.
        - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
        - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
        
        Returns:
        - tuple: Returns cartesian candidates and tree candidates.
        """
        # import pdb;pdb.set_trace()
        # Greedy decoding: Select the most probable candidate from the original logits.
        candidates_logit = paddle.argmax(logits[:, -1]).unsqueeze(0)
        TOPK = 10

        # import pdb;pdb.set_trace()
        # Extract the TOPK candidates from the medusa logits.
        candidates_medusa_logits = paddle.topk(medusa_logits[:, 0, -1], TOPK, dim = -1).indices

        # import pdb;pdb.set_trace()
        # Combine the selected candidate from the original logits with the topk medusa logits.
        candidates = paddle.cat([candidates_logit, candidates_medusa_logits.reshape_([-1])], dim=-1)

        # Map the combined candidates to the tree indices to get tree candidates.
        tree_candidates = candidates[tree_indices]

        # Extend the tree candidates by appending a zero.
        tree_candidates_ext = paddle.cat([tree_candidates, paddle.zeros((1), dtype='int64', device=tree_candidates.device)], dim=0)

        # Retrieve the cartesian candidates using the retrieve indices.
        cart_candidates = tree_candidates_ext[retrieve_indices]

        # import pdb;pdb.set_trace()
        # Unsqueeze the tree candidates for dimension consistency.
        tree_candidates = tree_candidates.unsqueeze(0)
        return cart_candidates, tree_candidates

    def sample(
        self,
        eos_token_id,
        top_k,
        top_p,
        penalty_score,
        frequency_score,
        presence_score,
        temperature=None,
        min_tokens_to_keep=1,
        **model_kwargs
    ):
        # print("keys: ", model_kwargs.keys())
        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(**args)
            return self(**model_inputs)

        def _post_process_(candidates, best_candidate, accept_length, tree_candidates, cache_k, cache_v, medusa_k, medusa_v, retrieve_indices):
            # 根据candidates 拉取next tokens
            next_tokens = candidates[:,best_candidate][:accept_length]

            # 确定是否停止
            length_cond = paddle.greater_equal(model_kwargs["step_idx"], model_kwargs["max_dec_len"])
            stop_flags = paddle.logical_or(model_kwargs["stop_flags"], length_cond)

            # TODO(@wufeisheng): 这里也要根据新的next_token重新设计
            set_stop_value_multi_ends_v2(
                next_tokens, stop_flags, model_kwargs["seq_lens_this_time"], eos_token_id
            )  # multi ends
            paddle.assign(stop_flags, model_kwargs["stop_flags"])

            # TODO(@wufeisheng): 实现新的saveoutput 因为encoder不输出token，deocder输出多个token
            save_output(next_tokens, model_kwargs["not_need_stop"], self.config.tensor_parallel_rank)

            # TODO(@wufeisheng): cache_k cache_v的更新
            update_cachekv(cache_k, cache_v, medusa_k, medusa_v, best_candidate, accept_length, tree_candidates, retrieve_indices)

            return next_tokens

        # Tree decoding
        tree_orig_logits, tree_combined_medusa_logits, padding_offset, cum_offsets = _forward_(**model_kwargs)  # [token_num, vocab_size]

        # tree_orig_logits [token_num, vocab_size] -> [num_decoders, num_posterior, self.medusa+1, vocab_size]  对于decoder需要确定某一个路径
        # -> [num_encoders, vocab_size] 对于encoder拉取对应的logits
        # 这里产生一个[num_decoders] 和 [num_encoders] 的索引 用于在[bsz]中填充

        # tree_combined_medusa_logits [4, token_num, vocab_size]也要变成 [4, num_decoders,medusa_len, vocab_size] [4, num_encoders, vocab_size]

        logits_decoder_index,  logits_encoder_index, insert_index_decoder, insert_index_encoder = medusa_rebuild_logits(model_kwargs["seq_lens_encoder"],model_kwargs["seq_lens_decoder"], 
                                                            padding_offset, model_kwargs["input_ids"], model_kwargs["retrieve_indices"])

        orig_logits_decoder,  orig_logits_encoder= tree_orig_logits[logits_decoder_index], tree_orig_logits[logits_encoder_index]
        orig_logits_decoder = orig_logits_decoder[:, model_kwargs["retrieve_indices"]]
        combined_medusa_logits_decoder, combined_medusa_logits_encoder = tree_combined_medusa_logits[:,logits_decoder_index], tree_combined_medusa_logits[:,logits_encoder_index]


        # combined_medusa_logits [4, num_decoders, medusa_len, vocab_size] 在路径选择结束后变成 [4, num_decoders, vocab_size]

        # Evaluate posterior
        # model_kwargs["medusa_candidates"] : [bsz, num_posterior, self.medusa+1]
        # 这里需要返回一个[4, num_decoders]的索引best_medusa_token_index，选出1/medusa 的那个token
        best_candidate, accept_length, best_medusa_token_index = self.evaluate_posterior(orig_logits_decoder, model_kwargs["medusa_candidates"], temperature, 0.09, 0.3)
        # best_candidate [num_decoders]
        combined_medusa_logits_decoder = combined_medusa_logits_decoder[best_medusa_token_index].squeeze()


        # 这里还要做一些事情 比如 流式输出, KV 拼接
        next_tokens = _post_process_(model_kwargs["medusa_candidates"], best_candidate, accept_length, tree_candidates, model_kwargs["cache_k"], model_kwargs["cache_v"], model_kwargs["medusa_k"], model_kwargs["medusa_v"])

        # Get candidates
        combined_medusa_logit = paddle.ones(shape=[4, bsz, vocab_size], dtype=orig_logits_decoder.dtype)
        combined_medusa_logit[:, insert_index_encoder] = combined_medusa_logits_encoder
        combined_medusa_logit[:, insert_index_decoder] = combined_medusa_logits_decoder
        orig_logits = paddle.ones(shape=[bsz, vocab_size], dtype=orig_logits_decoder.dtype)
        orig_logits[insert_index_encoder] = orig_logits_encoder
        orig_logits[insert_index_decoder] = orig_logits_decoder

        model_kwargs["medusa_candidates"], tree_candidates = self.generate_candidates(combined_medusa_logit, orig_logits, model_kwargs["tree_indices"], model_kwargs["retrieve_indices"])

        # update input_ids 和其他变量
        # input_ids 对于decoder来说就是tree_candidates, 对于encoder来说就是正常的输入
        # TODO(@wufeisheng): 实现新的update
        update_inputs(
            tree_candidates,
            model_kwargs["stop_flags"],
            model_kwargs["not_need_stop"],
            model_kwargs["seq_lens_this_time"],
            model_kwargs["seq_lens_encoder"],
            model_kwargs["seq_lens_decoder"],
            model_kwargs["input_ids"],
            model_kwargs["stop_nums"],
            next_tokens,
        )

        
    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.set_value(state_dict["lm_head.weight"])
        self.llama.set_state_dict({k: state_dict[k] for k in state_dict.keys()})


class LlamaForMiniGPT4InferenceModel(LlamaForCausalLMInferenceModel):
    """
    This class is 99% like LlamaForCausalLMInferenceModel.
    Used only for miniGPT4's second part.
    """

    # This function corresponds to miniGPT4's second part, only used in miniGPT4.
    @paddle.no_grad()
    def generate_text_with_image_features(
        self,
        image_features: paddle.Tensor,
        first_input_ids: paddle.Tensor,
        second_input_ids: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        min_length=None,
        max_length=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        step_idx=None,
        stop_flags=None,
        tgt_ids=None,
        tgt_pos=None,
        tgt_generation_mask=None,
        pre_ids=None,
        stop_nums=None,
        cache_kvs=[],
        inputs_embeds=None,
        **generate_kwargs
    ) -> paddle.Tensor:

        first_embeds = self.llama.embed_tokens(first_input_ids)
        second_embeds = self.llama.embed_tokens(second_input_ids)
        image_features = paddle.cast(image_features, dtype=first_embeds.dtype)
        inputs_embeds = paddle.concat([first_embeds, image_features, second_embeds], axis=1)

        outputs = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            penalty_score=penalty_score,
            frequency_score=frequency_score,
            presence_score=presence_score,
            min_length=min_length,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            step_idx=step_idx,
            stop_flags=stop_flags,
            tgt_ids=tgt_ids,
            tgt_pos=tgt_pos,
            tgt_generation_mask=tgt_generation_mask,
            pre_ids=pre_ids,
            stop_nums=stop_nums,
            cache_kvs=cache_kvs,
        )
        return outputs

    # rewrite to_static function in generation_utils.py
    def to_static(self, output_path: str, config: dict):
        dtype = config.get("dtype", paddle.get_default_dtype())
        cache_kvs_shapes = self.get_cache_kvs_shape(self.config, max_length=config.get("max_length", None))
        input_spec = [
            paddle.static.InputSpec(
                shape=[None, None, None], dtype="float32", name="image_features"
            ),  # image_features
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="first_input_ids"),  # first_input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="second_input_ids"),  # second_input_ids
            paddle.static.InputSpec(shape=[None, None], dtype=dtype, name="attention_mask"),  # attention_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),  # position_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_encoder"),  # seq_len_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_decoder"),  # seq_len_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_ids"),  # tgt_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_pos"),  # tgt_pos
            paddle.static.InputSpec(
                shape=[None, 1, 1, None], dtype=dtype, name="tgt_generation_mask"
            ),  # tgt_generation_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            paddle.static.InputSpec(shape=[1], dtype="int64", name="stop_nums"),  # stop_nums
            [
                paddle.static.InputSpec(
                    shape=shape,
                    dtype=dtype,
                    name="cache_kvs_{}".format(i),
                )
                for i, shape in enumerate(cache_kvs_shapes)
            ],  # cache_kvs
        ]

        model = paddle.jit.to_static(self.generate_text_with_image_features, input_spec=input_spec)
        paddle.jit.save(model, output_path)
