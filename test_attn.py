import math
import paddle 
import paddle.distributed as dist
from paddle.framework import LayerHelper, in_dynamic_mode
from paddle.nn import Layer
from paddle.nn.initializer import Constant

def flash_attn_unpadded_with_mask(
    query,
    key,
    value,
    attn_mask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    scale,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
):
    r"""
    
    """
    if in_dynamic_mode():
        (
            result_attention,
            result_softmax,
        ) = paddle._C_ops.flash_attn_unpadded(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            fixed_seed_offset,
            attn_mask,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            dropout,
            causal,
            return_softmax,
            not training,
            rng_name,
        )
        return result_attention, result_softmax if return_softmax else None

    helper = LayerHelper('flash_attn_unpadded', **locals())
    dtype = helper.input_dtype(input_param_name='q')
    out = helper.create_variable_for_type_inference(dtype)
    softmax = helper.create_variable_for_type_inference(dtype)
    softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
    seed_offset = helper.create_variable_for_type_inference(paddle.int64)
    inputs = {
        'q': query,
        'k': key,
        'v': value,
        'attn_mask': attn_mask,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_k,
        'fixed_seed_offset': fixed_seed_offset,
    }
    outputs = {
        'out': out,
        'softmax': softmax,
        'softmax_lse': softmax_lse,
        'seed_offset': seed_offset,
    }
    helper.append_op(
        type='flash_attn_unpadded',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_k': max_seqlen_k,
            'scale': scale,
            'dropout': dropout,
            'causal': causal,
            'return_softmax': return_softmax,
            'is_test': not training,
            'rng_name': rng_name,
        },
    )
    return out, softmax if return_softmax else None


token_num = 128
num_head = 40
dim_head = 128

cu_seqlens_q = paddle.to_tensor([100, 120, 125, 128])
cu_seqlens_k = paddle.to_tensor([100, 120, 125, 128])

bs=4

max_seqlen_q = 1024
max_seqlen_k = 1024

q = paddle.ones([token_num, num_head, dim_head], dtype='float16')
k = paddle.ones([token_num, num_head, dim_head], dtype='float16')
v = paddle.ones([token_num, num_head, dim_head], dtype='float16')

attn_mask = paddle.ones([bs, 1, 1024, 1024], dtype = 'float16')

scale = 1/math.sqrt(dim_head)

out = flash_attn_unpadded_with_mask(q, k, v, attn_mask, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale)

print(out)