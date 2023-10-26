from paddlenlp_ops import medusa_qkv_transpose_split_fetch_concat
import numpy as np
import paddle

medusa_k = paddle.to_tensor(np.load("npy/medusa_k.npy"))
medusa_v = paddle.to_tensor(np.load("npy/medusa_v.npy"))
qkv_out = paddle.to_tensor(np.load("npy/qkv_out.npy"))
block_tables = paddle.to_tensor(np.load("npy/block_tables.npy"))
cache_k = paddle.to_tensor(np.load("npy/cache_k.npy"))
cache_v = paddle.to_tensor(np.load("npy/cache_v.npy"))
seq_lens_encoder = paddle.to_tensor(np.load("npy/seq_lens_encoder.npy"))
seq_lens_decoder = paddle.to_tensor(np.load("npy/seq_lens_decoder.npy"))
cu_seqlens_q = paddle.to_tensor(np.load("npy/cu_seqlens_q.npy"))
cu_seqlens_k = paddle.to_tensor(np.load("npy/cu_seqlens_k.npy"))
padding_offsets = paddle.to_tensor(np.load("npy/padding_offsets.npy"))
input_ids = paddle.to_tensor(np.load("npy/input_ids.npy"))

print(qkv_out.shape)
print(input_ids.shape)
print(block_tables.shape)

q_out, k_out, v_out = medusa_qkv_transpose_split_fetch_concat(medusa_k, 
                                                                medusa_v,  
                                                                qkv_out, 
                                                                block_tables, 
                                                                cache_k, 
                                                                cache_v, 
                                                                seq_lens_encoder, 
                                                                seq_lens_decoder, 
                                                                cu_seqlens_q, 
                                                                cu_seqlens_k,
                                                                padding_offsets, 
                                                                input_ids,0)