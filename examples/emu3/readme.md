# Training Emu3-Gen for video generation in Megatron-LM
## Weight-conversion
```python
python tools/checkpoint/convert.py --model-type GPT --loader emu3 --saver mcore --load-dir /data/models/Emu3-Gen --save-dir /data/models/Emu3-Gen-Megatron --model-size emu3-gen --make-vocab-size-divisible-by 2 --saver-transformer-impl transformer_engine
```
## TE tf32 issue
since TE use tf32 by default, you may encoutered issue that the output is not numerically identical to torch's. See this issue https://github.com/NVIDIA/TransformerEngine/issues/1360 for more details.

## Model structure comparison

### Megatron
GPTModel(
  (embedding): LanguageModelEmbedding(
    (word_embeddings): VocabParallelEmbedding()
    (embedding_dropout): Dropout(p=0.0, inplace=False)
  )
  (rotary_pos_emb): RotaryEmbedding()
  (decoder): TransformerBlock(
    (layers): ModuleList(
      (0-31): 32 x TransformerLayer(
        (input_layernorm): IdentityOp()
        (self_attention): SelfAttention(
          (core_attention): TEDotProductAttention(
            (flash_attention): FlashAttention()
            (fused_attention): FusedAttention()
            (unfused_attention): UnfusedDotProductAttention(
              (scale_mask_softmax): FusedScaleMaskSoftmax()
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (linear_proj): TERowParallelLinear()
          (linear_qkv): TELayerNormColumnParallelLinear()
          (q_layernorm): IdentityOp()
          (k_layernorm): IdentityOp()
        )
        (pre_cross_attn_layernorm): IdentityOp()
        (cross_attention): IdentityOp()
        (cross_attn_bda): IdentityFuncOp()
        (pre_mlp_layernorm): IdentityOp()
        (mlp): MLP(
          (linear_fc1): TELayerNormColumnParallelLinear()
          (linear_fc2): TERowParallelLinear()
        )
      )
    )
    (final_layernorm): RMSNorm()
  )
  (output_layer): ColumnParallelLinear()
)
### Huggingface
Emu3ForCausalLM(
  (model): Emu3Model(
    (dropout): Dropout(p=0.1, inplace=False)
    (embed_tokens): Embedding(184622, 4096, padding_idx=151643)
    (layers): ModuleList(
      (0-31): 32 x Emu3DecoderLayer(
        (dropout): Dropout(p=0.1, inplace=False)
        (self_attn): Emu3SdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): Emu3RotaryEmbedding()
        )
        (mlp): Emu3MLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Emu3RMSNorm()
        (post_attention_layernorm): Emu3RMSNorm()
      )
    )
    (norm): Emu3RMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=184622, bias=False)
)