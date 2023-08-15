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


import numpy as np

import paddle
from paddle import _C_ops
from paddle.fluid import core
from paddle.fluid.core import VarDesc
from paddle.fluid.dygraph import no_grad
from paddle.fluid.framework import convert_np_dtype_to_dtype_
from paddle.incubate.nn import functional as incubate_f
from paddle.nn import Layer
from paddle.nn.initializer import Constant
from paddle.nn.layer.transformer import (
    _convert_attention_mask,
    _convert_param_attr_to_list,
)
from paddle.incubate.nn.functional import (
    variable_length_memory_efficient_attention,
    masked_multihead_attention, 
    fused_layer_norm,
    fused_rms_norm)
import paddle.distributed as dist
from paddlenlp_ops import rebuild_padding, write_cache_kv, qkv_transpose_split, rotary_qk, transpose_remove_padding, quant_int8, dequant_int8
from paddle.framework import (
    LayerHelper,
    convert_np_dtype_to_dtype_,
    core,
    in_dynamic_mode,
)
import numpy as np


# for distributed tensor model parallel
def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    if not in_dynamic_mode():
        # NOTE: use current_block and find_var_recursive to support while_loop
        startup_block = paddle.static.default_startup_program().current_block()
        main_block = paddle.static.default_main_program().current_block()
        startup_block._find_var_recursive(var.name).is_distributed = True
        main_block._find_var_recursive(var.name).is_distributed = True


# def masked_multihead_attention(
#     x,
#     cache_kv=None,
#     src_mask=None,
#     cum_offsets=None,
#     sequence_lengths=None,
#     rotary_tensor=None,
#     beam_cache_offset=None,
#     qkv_out_scale=None,
#     out_linear_shift=None,
#     out_linear_smooth=None,
#     seq_len=1,
#     rotary_emb_dims=0,
#     use_neox_rotary_style=False,
#     out_linear_in_scale=-1,
#     quant_round_type=1,
#     quant_max_bound=127.0,
#     quant_min_bound=-127.0,
# ):
#     r"""
#     Multi-head attention for text summarization.
#     This is a fusion operator to compute masked multihead attention in transformer model architecture.
#     This operator only supports running on GPU. The function of the transformer layer is consistent
#     with the following pseudo code:

#         .. code-block:: python

#             x = paddle.transpose(x, [0, 2, 1, 3])  # [batch\_size, sequence_length, num\_head, dim\_head] --> [batch\_size, num\_head, sequence_length, dim\_head]
#             q, k, v = paddle.split(x, 3, axis=2)
#             cache_k, cache_v= paddle.split(cache_kv_out, 2, axis=0)
#             k = paddle.concat([cache_k.squeeze(0), k], axis=2)
#             v = paddle.concat([cache_v.squeeze(0), v], axis=2)

#             product = paddle.matmul(x=q * (x.shape[3]**-0.5), y=k, transpose_y=True)
#             product = product + src_mask
#             product = paddle.nn.functional.softmax(product)
#             out = paddle.matmul(product, v).transpose([0, 2, 1, 3])

#     Args:
#         x (Tensor): The input tensor could be 4-D tensor, the input data type could be float16 or float32, the shape is `[batch\_size, 3, num\_head, dim\_head]`.
#         cache_kvs (list(Tensor)|tuple(Tensor)): The cache structure tensors for the generation model, the shape is `[2, batch\_size, num\_head, max\_seq\_len, head\_dim]`.
#         src_mask (Tensor): The src_mask tensor, the shape is `[batch\_size, 1, 1, sequence\_length]`.
#         sequence_lengths (Tensor, optional): The sequence_lengths tensor, the shape is `[batch\_size, 1]`.
#         rotary_tensor (Tensor, optional): The rotary_tensor tensor, the dtype must be float. the shape is `[batch\_size, 1, 1, sequence\_length, dim\_head]`.
#         beam_cache_offset (Tensor, optional): The beam_cache_offset tensor, the shape is `[batch\_size, beam\_size, max\_seq\_len + max\_dec\_len]`.
#         qkv_out_scale (Tensor, optional): The qkv_out_scale tensor, the shape is `[3, num\_head, dim\_head]]`.
#         out_linear_shift (Tensor, optional): The out_linear_shift tensor.
#         out_linear_smooth (Tensor, optional): The out_linear_smooth tensor.
#         beam_size (int, optional): The beam_size of beam search. Default 1.
#         rotary_emb_dims (int, optional): The rotary_emb_dims. Default 0.
#         use_neox_rotary_style (bool, optional): A flag indicating whether neox_rotary_style is needed or not. Default False.
#         out_linear_in_scale (float, optional): The out_linear_in_scale.
#         quant_round_type (int, optional): The quant_round_type. Default 1.
#         quant_max_bound (float, optional): The quant_max_bound. Default 127.0.
#         quant_min_bound (float, optional): The quant_min_bound. Default -127.0.

#     Returns:
#         Tensor|tuple: If "beam_cache_offset_out" is not none, return the
#         tuple (output, cache_kvs_out, beam_cache_offset_out), which output is the output of
#         masked_multihead_attention layers, cache_kvs_out is inplace with input `cache_kvs`.
#         If "beam_cache_offset_out" is none, return the tuple (output, cache_kvs_out).

#     Examples:
#         .. code-block:: python

#             # required: gpu
#             import paddle
#             import paddle.incubate.nn.functional as F

#             # input: [batch_size, 3, num_head, dim_head]
#             x = paddle.rand(shape=(2, 3, 32, 128), dtype="float32")

#             # src_mask: [batch_size, 1, 1, sequence_length]
#             src_mask = paddle.rand(shape=(2, 1, 1, 10), dtype="float32")

#             # cache_kv: [2, batch_size, num_head, max_seq_len, dim_head]
#             cache_kv = paddle.rand(shape=(2, 2, 32, 64, 128), dtype="float32")

#             output = F.masked_multihead_attention(
#                 x, src_mask=src_mask, cache_kv=cache_kv)

#     """

#     if in_dynamic_mode():
#         return _C_ops.masked_multihead_attention_(
#             x,
#             cache_kv,
#             src_mask,
#             cum_offsets,
#             sequence_lengths,
#             rotary_tensor,
#             beam_cache_offset,
#             qkv_out_scale,
#             out_linear_shift,
#             out_linear_smooth,
#             seq_len,
#             rotary_emb_dims,
#             use_neox_rotary_style,
#             out_linear_in_scale,
#             quant_round_type,
#             quant_max_bound,
#             quant_min_bound,
#         )

#     helper = LayerHelper('masked_multihead_attention_', **locals())
#     if out_linear_in_scale > 0:
#         out = helper.create_variable_for_type_inference(dtype='int8')
#     else:
#         out = helper.create_variable_for_type_inference(dtype=x.dtype)


#     inputs = {}
#     inputs['x'] = x
#     inputs['cache_kv'] = cache_kv
#     if src_mask is not None:
#         inputs['src_mask'] = src_mask
#     if cum_offsets is not None:
#         inputs['cum_offsets'] = cum_offsets
#     if sequence_lengths is not None:
#         inputs['sequence_lengths'] = sequence_lengths
#     if rotary_tensor is not None:
#         inputs['rotary_tensor'] = rotary_tensor
#     beam_cache_offset_flag = False
#     if beam_cache_offset is not None:
#         inputs['beam_cache_offset'] = beam_cache_offset
#         beam_cache_offset_flag = True
#     else:
#         beam_cache_offset = helper.create_variable_for_type_inference(
#             dtype="int"
#         )
#     if qkv_out_scale is not None:
#         inputs['qkv_out_scale'] = qkv_out_scale
#     if out_linear_shift is not None:
#         inputs['out_linear_shift'] = out_linear_shift
#     if out_linear_smooth is not None:
#         inputs['out_linear_smooth'] = out_linear_smooth

#     outputs = {
#         'out': out,
#         'cache_kv_out': cache_kv,
#         'beam_cache_offset_out': beam_cache_offset,
#     }
#     helper.append_op(
#         type='masked_multihead_attention',
#         inputs=inputs,
#         outputs=outputs,
#         attrs={
#             'seq_len': seq_len,
#             'rotary_emb_dims': rotary_emb_dims,
#             'use_neox_rotary_style': use_neox_rotary_style,
#             'out_scale': out_linear_in_scale,
#             'quant_round_type': quant_round_type,
#             'quant_max_bound': quant_max_bound,
#             'quant_min_bound': quant_min_bound,
#         },
#     )
#     return (
#         (out, cache_kv, beam_cache_offset)
#         if beam_cache_offset_flag is not None
#         else (out, cache_kv)
#     )


def fused_act_bias_wrapper(
    x,
    bias=None,
    dequant_scales=None,
    shift=None,
    smooth=None,
    act_method='gelu',
    compute_dtype='default',
    quant_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    if in_dynamic_mode():
        return paddle._C_ops.fused_bias_act(
            x,
            bias,
            dequant_scales,
            shift,
            smooth,
            act_method,
            compute_dtype,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    else:
        helper = LayerHelper("fused_bias_act")
        if (x.dtype == "int32"):
            if (compute_dtype == "bf16"):
                dtype = "uint16"
            elif (compute_dtype == "fp16"):
                dtype = "float16"
            elif (compute_dtype == "fp32"):
                dtype = "float32"
            out = helper.create_variable_for_type_inference(dtype=dtype)
        else:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        inputs={
            'x': x,
            'bias': bias,
            'dequant_scales': dequant_scales,
            'shift': shift,
            'smooth': smooth,
        }
        attrs={
            'act_method': act_method,
            'compute_dtype': compute_dtype,
            'quant_scale': quant_scale,
            'quant_round_type': quant_round_type,
            'quant_max_bound': quant_max_bound,
            'quant_min_bound': quant_min_bound
        }

        helper.append_op(
            type="fused_bias_act",
            inputs=inputs,
            outputs={'out': out},
            attrs=attrs,
        )
        return out


def matmul_int8(x, y, transpose_x, transpose_y):
    if in_dynamic_mode():
        return paddle._C_ops.matmul_int8(x, y, transpose_x, transpose_y)
    else:
        helper = LayerHelper("matmul_int8")
        out = helper.create_variable_for_type_inference(dtype='int32')

        inputs={
            'x': x,
            'y': y
        }
        attrs={
            'transpose_x': transpose_x,
            'transpose_y': transpose_y
        }

        helper.append_op(
            type="matmul_int8",
            inputs=inputs,
            outputs={'out': out},
            attrs=attrs,
        )
        return out

class SwiGLU(Layer):
    def __init__(self):
        super().__init__()
        self.act = paddle.nn.Silu()

    def forward(self, x, gate):
        return self.act(x) * gate



class FusedMultiTransformerInt8(Layer):
    """
    FusedMultiTransformer is composed of multi transformer layers which contains two
    sub-layers which are self (multi-head) attention and feedforward network. The
    function of one transformer layer is consistent with the following pseudo code:

    .. code-block:: python

        if pre_layer_norm:
            out = layer_norm(x)
            out = qkv_linear(out) + qkv_bias
        else:
            out = qkv_linear(x) + qkv_bias
        out = transpose(out, perm=[2, 0, 3, 1, 4])
        # extract q, k and v from out.
        q = out[0:1, ::]
        k = out[1:2, ::]
        v = out[2:3, ::]
        out = q * k^t
        out = attn_mask + out
        out = softmax(out)
        out = dropout(out)
        out = out * v
        out = transpose(out, perm=[0, 2, 1, 3])
        out = linear(out)
        if pre_layer_norm:
            out = x + dropout(out + bias)
        else:
            out = layer_norm(x + dropout(out + bias))

        residual = out;
        if pre_layer_norm:
            out = ffn_layer_norm(out)
        out = ffn1_linear(out)
        out = dropout(activation(out + ffn1_bias))
        out = ffn2_linear(out)
        out = residual + dropout(out + ffn2_bias)
        if not pre_layer_norm:
            out = ffn_layer_norm(out)

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.0
        activation (str, optional): The activation function in the feedforward
            network. Default "gelu".
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default True
        ln_scale_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for Attention layer_norm. For Attention layer_norm weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for Attention layer_norm. For Attention layer_norm bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        qkv_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for Attention qkv computation. For Attention qkv weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        qkv_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for Attention qkv computation. For Attention qkv bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for Attention linear. For Attention linear weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for Attention linear computation. For Attention linear bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn_ln_scale_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for FFN layer_norm. For FFN layer_norm weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn_ln_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for FFN layer_norm. For FFN layer_norm bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn1_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for FFN first linear. For FFN first linear weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn1_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for FFN first linear. For FFN first linear bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn2_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for FFN second linear. For FFN second linear weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn2_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for FFN second linear. For FFN second linear bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1, etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        epsilon (float, optional): Small float value added to denominator of the layer_norm to
            avoid dividing by zero. Default: 1e-05.
        num_layers (int, optional): The number of layers of the transformer. If `qkv_weight_attrs`
            is a list or tuple, the number of layers is obtained from `qkv_weight_attrs`. num_layers
            only takes effect when `qkv_weight_attrs` is not a list or tuple. Default: -1.
        nranks (int, optional): Distributed tensor model parallel nranks. Default is 1, means not using mp.
        trans_qkvw (bool, optional): Whether to transpose for weights of qkv.
            If true, the shape eights of qkv should be [3, num_head, dim_head, dim_embed].
            Otherwise the shape of weights of qkv should be [dim_embed, 3, num_head, dim_head]. Default: True.
        ring_id (int, optional): For distributed tensor model parallel. Default is -1, means not using mp.
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedMultiTransformer

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, 1, src_len, src_len]
            attn_mask = paddle.rand((2, 1, 4, 4))
            encoder_layers = FusedMultiTransformer(128, 2, 512, num_layers=1)
            enc_output = encoder_layers(enc_input, attn_mask)  # [2, 4, 128]
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        norm_type="layernorm",
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        ffn1_weight_attrs=None,
        ffn1_bias_attrs=None,
        ffn2_weight_attrs=None,
        ffn2_bias_attrs=None,
        qkv_weight_out_scale_attrs=None,
        linear_weight_out_scale_attrs=None,
        ffn1_weight_out_scale_attrs=None,
        ffn2_weight_out_scale_attrs=None,
        linear_shift_attrs=None,
        linear_smooth_attrs=None,
        ffn2_shift_attrs=None,
        ffn2_smooth_attrs=None,
        epsilon=1e-5,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        name=None,
    ):
        super().__init__()

        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id
        self.nranks = nranks
        self.norm_type = norm_type
        if  norm_type == "layernorm":
            self.norm_func = fused_layer_norm
        else:
            self.norm_func = fused_rms_norm

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        assert dim_feedforward % nranks == 0
        num_heads = num_heads // nranks
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward

        if isinstance(qkv_weight_attrs, (list, tuple)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.ln_scales, self.ln_biases = [], []
        self.qkv_weights, self.qkv_weight_out_scales, self.qkv_biases = [], [], []
        self.linear_weights, self.linear_weight_out_scales, self.linear_biases = [], [], []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []
        self.ffn1_weights, self.ffn1_weight_out_scales, self.ffn1_biases = [], [], []
        self.ffn2_weights, self.ffn2_weight_out_scales, self.ffn2_biases = [], [], []

        self.linear_shifts,self.linear_smooths, self.ffn2_shifts, self.ffn2_smooths = [], [], [], []

        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs
        
        def _add_parameter(param):
            assert param.name not in self._parameters
            self._parameters[param.name] = param

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn1_bias_attr = get_attr(ffn1_bias_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)
            ffn2_bias_attr = get_attr(ffn2_bias_attrs, i)
            qkv_weight_out_scale_attr = get_attr(qkv_weight_out_scale_attrs,i)
            linear_weight_out_scale_attr = get_attr(linear_weight_out_scale_attrs,i)
            ffn1_weight_out_scale_attr = get_attr(ffn1_weight_out_scale_attrs,i)
            ffn2_weight_out_scale_attr = get_attr(ffn2_weight_out_scale_attrs,i)


            linear_shift_attr=get_attr(linear_shift_attrs, i)
            linear_smooth_attr=get_attr(linear_smooth_attrs, i)
            ffn2_shift_attr=get_attr(ffn2_shift_attrs, i)
            ffn2_smooth_attr=get_attr(ffn2_smooth_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
                dtype='float32',
            )
            ln_bias = None 
            if ln_bias_attr: 
              ln_bias = self.create_parameter(
                  attr=ln_bias_attr, shape=[embed_dim], is_bias=True,
                  dtype='float32',
              )
            qkv_weight = self.create_parameter(
                shape=[3 * num_heads * self.head_dim, embed_dim]
                if trans_qkvw
                else [embed_dim, 3 * num_heads * self.head_dim],
                attr=qkv_weight_attr,
                dtype='int8',
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0)
            )
            qkv_weight_out_scale = self.create_parameter(
                shape=[self.head_dim*3*num_heads],
                attr=qkv_weight_out_scale_attr,
                dtype='float32',
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0)
            )
            qkv_bias = None
            if qkv_bias_attr: 
              qkv_bias = self.create_parameter(
                  shape=[3 * num_heads * self.head_dim],
                  attr=qkv_bias_attr,
                  dtype=self._dtype,
                  is_bias=True,
              )
            linear_weight = self.create_parameter(
                shape=[embed_dim, num_heads * self.head_dim],
                attr=linear_weight_attr,
                dtype='int8',
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0)
            )
            linear_weight_out_scale = self.create_parameter(
                shape=[embed_dim],
                attr=linear_weight_out_scale_attr,
                dtype='float32',
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0)
            )
            linear_bias = None 
            if linear_bias_attr: 
              linear_bias = self.create_parameter(
                  shape=[embed_dim],
                  attr=linear_bias_attr,
                  dtype=self._dtype,
                  is_bias=True,
              )

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype='float32',
            )
            ffn_ln_bias = None 
            if ffn_ln_bias_attr: 
              ffn_ln_bias = self.create_parameter(
                  shape=[embed_dim], attr=ffn_ln_bias_attr, is_bias=True,
                  dtype='float32',
              )
            ffn1_weight = self.create_parameter(
                shape=[dim_feedforward * 2, embed_dim] if activation.endswith("glu") else [dim_feedforward, embed_dim],
                attr=ffn1_weight_attr,
                dtype='int8',
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0)
            )

            ffn1_weight_out_scale = self.create_parameter(
                shape=[dim_feedforward * 2] if activation.endswith("glu") else [dim_feedforward],
                attr=ffn1_weight_out_scale_attr,
                dtype='float32',
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0)
            )
            ffn1_bias = None 
            if ffn1_bias_attr: 
              ffn1_bias = self.create_parameter(
                  shape=[dim_feedforward * 2] if activation.endswith("glu") else [dim_feedforward],
                  attr=ffn1_bias_attr,
                  dtype=self._dtype,
                  is_bias=True,
              )
            ffn2_weight = self.create_parameter(
                shape=[embed_dim, dim_feedforward],
                attr=ffn2_weight_attr,
                dtype='int8',
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0)
            )
            ffn2_weight_out_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn2_weight_out_scale_attr,
                dtype='float32',
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0)
            )
            ffn2_bias = None 
            if ffn2_bias_attr: 
              ffn2_bias = self.create_parameter(
                  shape=[embed_dim],
                  attr=ffn2_bias_attr,
                  dtype=self._dtype,
                  is_bias=True,
              )

            linear_shift = self.create_parameter(
                shape=[num_heads * self.head_dim],
                attr=linear_shift_attr,
                dtype=self._dtype,
                is_bias=False)

            linear_smooth = self.create_parameter(
                shape=[num_heads * self.head_dim],
                attr=linear_smooth_attr,
                dtype=self._dtype,
                is_bias=False)

            ffn2_shift = self.create_parameter(
                shape=[dim_feedforward],
                attr=ffn2_shift_attr,
                dtype=self._dtype,
                is_bias=False)

            ffn2_smooth = self.create_parameter(
                shape=[dim_feedforward],
                attr=ffn2_smooth_attr,
                dtype=self._dtype,
                is_bias=False)

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_weight)
                _set_var_distributed(ffn1_bias)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_weight_out_scales.append(qkv_weight_out_scale)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_weight_out_scales.append(linear_weight_out_scale)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_weight_out_scales.append(ffn1_weight_out_scale)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_weight_out_scales.append(ffn2_weight_out_scale)
            self.ffn2_biases.append(ffn2_bias)

            self.linear_shifts.append(linear_shift)
            self.linear_smooths.append(linear_smooth)
            self.ffn2_shifts.append(ffn2_shift)
            self.ffn2_smooths.append(ffn2_smooth)

            _add_parameter(ln_scale)
            _add_parameter(ln_bias)
            _add_parameter(qkv_weight)
            _add_parameter(qkv_weight_out_scale)
            _add_parameter(qkv_bias)
            _add_parameter(linear_weight)
            _add_parameter(linear_weight_out_scale)
            _add_parameter(linear_bias)

            _add_parameter(ffn_ln_scale)
            _add_parameter(ffn_ln_bias)
            _add_parameter(ffn1_weight)
            _add_parameter(ffn1_weight_out_scale)
            _add_parameter(ffn1_bias)
            _add_parameter(ffn2_weight)
            _add_parameter(ffn2_weight_out_scale)
            _add_parameter(ffn2_bias)

            _add_parameter(linear_shift)
            _add_parameter(linear_smooth)
            _add_parameter(ffn2_shift)
            _add_parameter(ffn2_smooth)

        self.dropout_rate = dropout_rate
        self.activation = activation
        if self.activation == "swiglu":
            self.swiglu = SwiGLU()
        self.name = name

    def forward(
        self,
        input_ids,
        src,
        cum_offsets=None,
        padding_offset=None,
        attn_mask=None,
        caches=None,
        rotary_embs=None,
        rotary_emb_dims=0,
        seq_lens=None,
        time_step=None,
    ):
        if caches is not None:
            assert len(caches) == len(self.qkv_weights)

        residual_out = src
        for i in range(len(caches)):
            if i == 0:
                # TODO(wangbojun), need real scale for ptq
                ln_out = self.norm_func(
                    src, 
                    self.ln_scales[i],
                    self.ln_biases[i],
                    self._epsilon,
                    1.0,
                    begin_norm_axis=1,
                    quant_scale=self.act_scales['qkv_in_scale'][i], #quant_in_scale
                    quant_round_type=1, #quant_round_type
                    quant_max_bound=127.0, # quant_max_bound
                    quant_min_bound=-127.0 # quant_min_bound
                    )[0]

            qkv_out = matmul_int8(ln_out, self.qkv_weights[i], False, True)
            if time_step is None:
                # TODO(wangbojun), need dequant layer here for  context stage 
                qkv_out = dequant_int8(qkv_out, self.qkv_biases[i], 
                    self.qkv_weight_out_scales[i],
                    )
                pass
            else:
                # TODO(wangbojun), for generator stage, dequant is in mmha
                qkv_out = dequant_int8(qkv_out, self.qkv_biases[i], 
                    self.qkv_weight_out_scales[i],
                    )
                pass
            
            qkv_out = paddle.add(qkv_out, self.qkv_biases[i])
            # fmha compute
            if time_step is None: # context
                # qkv transpose split
                q_out, k_out, v_out = qkv_transpose_split(
                    qkv_out, 
                    padding_offset, 
                    seq_lens, 
                    input_ids,
                    self.num_heads // self.nranks,
                    self.head_dim)
                # rotary emb (inplace)
                tmp_out = rotary_qk(
                    q_out,
                    k_out,
                    rotary_embs,
                    seq_lens,
                    padding_offset,
                    input_ids,
                    rotary_emb_dims
                )
                # write cache kv (inplace)
                tmp_out = write_cache_kv(k_out, v_out, caches[i], seq_lens)

                fmha_out = variable_length_memory_efficient_attention(
                    q_out,
                    k_out,
                    v_out,
                    seq_lens,
                    seq_lens,
                    mask=attn_mask
                )
                fmha_out = transpose_remove_padding(fmha_out, seq_lens, padding_offset)
                fmha_out = fmha_out.reshape([-1, self.num_heads // self.nranks * self.head_dim])
    
                # out_linear
                fmha_out_in_scale = self.act_scales['out_linear_in_scale'][i]
                fmha_out = quant_int8(fmha_out, self.linear_shifts[i],  self.linear_smooths[i], fmha_out_in_scale,0,127.0,-127.0)
            else:
                fmha_out = masked_multihead_attention(
                    x=qkv_out,
                    cache_kv=caches[i],
                    src_mask=attn_mask,
                    sequence_lengths=seq_lens,
                    rotary_tensor=rotary_embs,
                    rotary_emb_dims=1,
                    beam_cache_offset=None, #beam_cache_offset
                    out_linear_shift=self.linear_shifts[i], #out_linear_shift
                    out_linear_smooth=self.linear_smooths[i], #out_linear_smooth,
                    out_linear_in_scale=self.act_scales['out_linear_in_scale'][i],
                    quant_round_type=0,
                    quant_max_bound=127.0,
                    quant_min_bound=-127.0
                )[0]
            fmha_out = fmha_out.reshape([-1, self.num_heads // self.nranks * self.head_dim])
            # out_linear
            out_linear_out = matmul_int8(fmha_out,self.linear_weights[i],False,True)
            out_linear_out = dequant_int8(out_linear_out, self.linear_biases[i], 
                self.linear_weight_out_scales[i]
                )

            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(out_linear_out)
            tmp_out, residual_out = self.norm_func(
                out_linear_out,
                self.ffn_ln_scales[i],
                self.ffn_ln_biases[i],
                self._epsilon,
                1.0,
                1,
                bias=self.linear_biases[i],
                residual=residual_out,
                quant_scale=self.act_scales['ffn1_in_scale'][i],
                quant_round_type=1, #quant_round_type
                quant_max_bound=127.0, # quant_max_bound
                quant_min_bound=-127.0 # quant_min_bound
            )[:2]

            ffn1_out = matmul_int8(tmp_out,self.ffn1_weights[i],False,True)
            ffn1_out = fused_act_bias_wrapper(
                    ffn1_out, 
                    self.ffn1_biases[i], 
                    self.ffn1_weight_out_scales[i], #dequant scales
                    act_method=self.activation,
                    compute_dtype='bf16',
                    shift=self.ffn2_shifts[i],
                    smooth=self.ffn2_smooths[i],
                    quant_scale=self.act_scales['ffn2_in_scale'][i], #quant_scale
                    quant_round_type=1, #quant_round_type
                    quant_max_bound=127.0, # quant_max_bound
                    quant_min_bound=-127.0 #quant_max_bound
                    )

            # ffn2 matmul
            # TODO(wangtbojun)
            ffn2_out = matmul_int8(ffn1_out, self.ffn2_weights[i] ,False,True)

            # TODO(need dequant)
            ffn2_out = dequant_int8(ffn2_out, self.ffn2_biases[i], 
            self.ffn2_weight_out_scales[i],
            )

            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(ffn2_out)

            # norm + residual_add_bias

            if i != len(caches) - 1:
                tmp_out,residual_out = self.norm_func(
                    ffn2_out,
                    self.ln_scales[i+1],
                    self.ln_biases[i+1],
                    self._epsilon,
                    1.0,
                    1,
                    bias=self.ffn2_biases[i],
                    residual=residual_out,
                    quant_scale=self.act_scales['qkv_in_scale'][i+1],
                    quant_round_type=1, #quant_round_type
                    quant_max_bound=127.0, # quant_max_bound
                    quant_min_bound=-127.0 # quant_min_bound
                )[:2]
            else:
                tmp_out, _, _, _ = fused_layer_norm(
                    ffn2_out,
                    None,
                    None,
                    self._epsilon,
                    bias=self.ffn2_biases[i],
                    residual=residual_out
                )
            ln_out = tmp_out
        if time_step is None:
            out = rebuild_padding(tmp_out, cum_offsets, seq_lens, input_ids)
        else:
            out = tmp_out
        return out