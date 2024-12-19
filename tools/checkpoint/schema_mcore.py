# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Mcore model schemas."""

import typing as T

from schema_base import ModelSchema


def get_mcore_transformer_block_key(model_key):
    return {
        "GPT" : "decoder",
        "BERT" : "encoder",
    }[model_key]


class MCoreSchema(ModelSchema):

    def __init__(self, model_type, layer_schema):
        block_key = get_mcore_transformer_block_key(model_type)
        super().__init__({
            "embeddings" : {
                "pos" : "embedding.position_embeddings.weight",
                "word" : "embedding.word_embeddings.weight",
            },
            "layer_prefix" : f"{block_key}.layers",
            "layer" : layer_schema,
            "final_norm" : {
                "weight" : f"{block_key}.final_layernorm.weight",
                "bias" : f"{block_key}.final_layernorm.bias",
            },
            "output_layer" : {
                "weight" : "output_layer.weight",
            },
            "pooler" : {
                "weight" : "pooler.dense.weight",
                "bias" : "pooler.dense.bias",
            },
            "lm_head" : {
                "dense_weight" : "lm_head.dense.weight",
                "dense_bias" : "lm_head.dense.bias",
                "norm_weight" : "lm_head.layer_norm.weight",
                "norm_bias" : "lm_head.layer_norm.bias",
            },
            "binary_head" : {
                "weight" : "binary_head.weight",
                "bias" : "binary_head.bias",
            },
        })


class MCoreLocalSchema(MCoreSchema):

    def __init__(self, model_type):
        super().__init__(model_type, layer_schema={
            # Self attention.
            "self_attn_norm_weight" : "input_layernorm.weight",
            "self_attn_norm_bias" : "input_layernorm.bias",
            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",
            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",

            # MLP.
            "mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "pre_mlp_layernorm.bias",
            "mlp_fc1_weight" : "mlp.linear_fc1.weight",
            "mlp_fc1_bias" : "mlp.linear_fc1.bias",
            "mlp_fc2_weight" : "mlp.linear_fc2.weight",
            "mlp_fc2_bias" : "mlp.linear_fc2.bias",

        })


class MCoreTESchema(MCoreSchema):

    def __init__(self, model_type, qkv_linear_fusion):
        if qkv_linear_fusion:
            layer_schema = {
                # qkv proj and norm
                "self_attn_norm_weight" : "self_attention.linear_qkv.layer_norm_weight",
                "self_attn_norm_bias" : "self_attention.linear_qkv.layer_norm_bias",
                "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
                "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",
            }
        else:
            layer_schema = {
                # qkv proj and norm
                "self_attn_norm_weight" : "input_layernorm.weight",
                "self_attn_norm_bias" : "input_layernorm.bias",
                "self_attn_q_weight": "self_attention.linear_q.weight",
                "self_attn_k_weight": "self_attention.linear_k.weight",
                "self_attn_v_weight": "self_attention.linear_v.weight",
                "self_attn_q_bias": "self_attention.linear_q.bias",
                "self_attn_k_bias": "self_attention.linear_k.bias",
                "self_attn_v_bias": "self_attention.linear_v.bias",
            }
        print(layer_schema)
        layer_schema |= {
            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",
            # MLP.
            "mlp_norm_weight" : "mlp.linear_fc1.layer_norm_weight",
            "mlp_norm_bias" : "mlp.linear_fc1.layer_norm_bias",
            "mlp_fc1_weight" : "mlp.linear_fc1.weight",
            "mlp_fc1_bias" : "mlp.linear_fc1.bias",
            "mlp_fc2_weight" : "mlp.linear_fc2.weight",
            "mlp_fc2_bias" : "mlp.linear_fc2.bias"
        }
        print(layer_schema)
        super().__init__(model_type, layer_schema=layer_schema)


class MCoreMoETESchema(MCoreSchema):

    def __init__(self, model_type, num_experts, expert_model_parallel_size):
        num_local_experts = num_experts // expert_model_parallel_size
        super().__init__(model_type, layer_schema={
            # Self attention.
            "self_attn_norm_weight" : "self_attention.linear_qkv.layer_norm_weight",
            "self_attn_norm_bias" : "self_attention.linear_qkv.layer_norm_bias",

            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",

            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",

            # MLP.
            "mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "pre_mlp_layernorm.bias",

            "router_weight" : "mlp.router.weight",

            **{f"mlp_fc1_weight.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.linear_fc1.weight" for expert_idx in range(num_local_experts) },
            **{f"mlp_fc2_weight.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.linear_fc2.weight" for expert_idx in range(num_local_experts) },

        })


def get_model_schema(
    model_type: T.Literal["GPT", "BERT"],
    transformer_impl: T.Literal["transformer_engine", "local"],
    qkv_linear_fusion: bool = True,
    num_experts: T.Optional[int] = None,
    expert_model_parallel_size: T.Optional[int] = None,
) -> MCoreSchema:
    # Handle MoE case first
    if num_experts is not None and num_experts > 0:
        assert transformer_impl == "transformer_engine", "MoE only supports transformer_engine implementation"
        assert isinstance(expert_model_parallel_size, int), "expert_model_parallel_size must be specified for MoE"
        return MCoreMoETESchema(model_type, num_experts, expert_model_parallel_size)

    # Handle local implementation
    if transformer_impl == "local":
        assert qkv_linear_fusion, "local implementation only supports qkv_linear_fusion=True"
        return MCoreLocalSchema(model_type)

    # Handle transformer_engine implementation
    assert transformer_impl == "transformer_engine"
    return MCoreTESchema(model_type, qkv_linear_fusion)
