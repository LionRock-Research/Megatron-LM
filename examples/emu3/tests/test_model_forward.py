import torch
import gc
import os
import numpy as np
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.training.checkpointing import _load_base_checkpoint
from types import SimpleNamespace
from transformers import AutoModelForCausalLM

EMU3_HF_MODEL_DIR = "/data/models/Emu3-Gen"
EMU3_MEGATRON_MODEL_DIR = "/data/models/Emu3-Gen-Megatron"
CACHE_DIR = "/root/Megatron-LM/cache"
SEQ_LEN = 128
SEED = 42
ATOL = {
    'float32': 1e-3,
    'bfloat16': 1e-1,
    'float16': 5e-2,
}
RTOL = {
    'float32': 1.3e-6,
    'bfloat16': 1e-2,
    'float16': 1e-2,
}
DTYPE_DICT = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}
DTYPE = 'bfloat16'

def check_tol(t1, t2, atol=5e-4, rtol=1.3e-6):
    result = torch.allclose(t1.float(), t2.float(), atol=atol, rtol=rtol)
    if not result:
        diff = torch.abs(t1 - t2).flatten()
        m = torch.argmax(diff)
        msg = (
            f"Outputs not close enough in tensor. "
            f"Location of the maximum difference: {m.item()} "
            f"with {t1.flatten()[m].item()} vs {t2.flatten()[m].item()} "
            f"(diff {diff[m].item()})."
            f"mean diff: {diff.mean().item()}."
            f"min diff: {diff.min().item()}."
        )
        print(msg)
    return result

def generate_hook(module, suffix, dump_input=False):
    def debug_hook(module, input, output):
        def dump_tensor(tensor_to_be_dumped, output_file_name):
            if isinstance(tensor_to_be_dumped, tuple):
                for i, o in enumerate(tensor_to_be_dumped):
                    try:
                        if torch.is_tensor(o):
                            np.save(f"{output_file_name}_o{i}.npy", o.clone().detach().to("cpu").float().numpy())
                        else:
                            print(f"Skipping non-tensor output {i} for {suffix}")
                    except Exception as e:
                        print(f"Error saving {suffix}'s output {i}: {e}")
            else:
                try:
                    if torch.is_tensor(tensor_to_be_dumped):
                        np.save(f"{output_file_name}.npy", tensor_to_be_dumped.clone().detach().to("cpu").float().numpy())
                    else:
                        print(f"Skipping non-tensor output for {suffix}")
                except Exception as e:
                    print(f"Error saving {suffix}'s output: {e}")

        os.makedirs(CACHE_DIR, exist_ok=True)
        
        if dump_input:
            output_file_name = f"{CACHE_DIR}/{suffix}_input"
            dump_tensor(input, output_file_name)
        # dump output
        output_file_name = f"{CACHE_DIR}/{suffix}"
        dump_tensor(output, output_file_name)

    return debug_hook


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)


def destroy_distributed():
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

def load_weights(gpt_model):
    args = SimpleNamespace(
        load=EMU3_MEGATRON_MODEL_DIR,
        non_persistent_global_ckpt_dir=None,
        exit_on_missing_checkpoint=True,
        non_persistent_ckpt_type="global",
    )
    state_dict, _, _, _ = _load_base_checkpoint(
        load_dir=args.load,
        args=args,
        rank0=False,
        checkpointing_context=None,
    )
    gpt_model.load_state_dict(state_dict["model"])
    gpt_model.cuda()
    torch.cuda.empty_cache()
    gc.collect()


def set_deterministic_mode():
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_model_forward_and_backward(debug=False, deterministic_mode=False, test_backward=False):
    # set deterministic mode
    if deterministic_mode:
        set_deterministic_mode()

    initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(SEED)
    # prepare input
    embed_grad_hf = torch.randint(0, 184622, (1, SEQ_LEN), dtype=torch.int32, device="cuda")
    input_megatron = embed_grad_hf.clone().detach()
    position_ids_hf = torch.arange(SEQ_LEN, dtype=torch.int32, device="cuda").unsqueeze(0)
    position_ids_megatron = position_ids_hf.clone().detach()
    # Create causal attention mask where upper triangle is 0 and lower triangle is -inf
    attention_mask_megatron = torch.triu(torch.ones((SEQ_LEN, SEQ_LEN)), diagonal=1).bool()
    attention_mask_megatron = attention_mask_megatron.unsqueeze(0).unsqueeze(0)
    attention_mask_megatron = attention_mask_megatron.to(device="cuda")
    attention_mask_megatron = torch.zeros_like(attention_mask_megatron, dtype=torch.float32).masked_fill(attention_mask_megatron, float("-inf"))
    # test megatron model
    print("Building megatron model...")
    transformer_config = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        attention_softmax_in_fp32=True,
        bf16=DTYPE == 'bfloat16',
        fp16=DTYPE == 'float16',
        ffn_hidden_size=14336,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=True,
        add_bias_linear=False,
        deterministic_mode=deterministic_mode,
        params_dtype=DTYPE_DICT[DTYPE],
    )
    print(transformer_config)
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=184622,
        max_sequence_length=9216,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rope_scaling=False,
        rotary_base=1000000.0,
        pre_process=True,
        post_process=True
    )
    print(gpt_model)
    load_weights(gpt_model)
    print("Megatron model built.")
    print(gpt_model.decoder.layers[0].self_attention.linear_qkv.layer_norm_weight.detach().cpu().float().numpy())
    print("Running megatron model forward and backward...")
    gpt_model.eval()  # Switch to train mode
    if debug:
        for name, module in gpt_model.named_modules():
            module.register_forward_hook(generate_hook(module, "megatron_" + name))
        # rms norm weight
        np.save(f"{CACHE_DIR}/megatron_rmsnorm_weight.npy", gpt_model.decoder.layers[0].self_attention.linear_qkv.layer_norm_weight.clone().detach().to("cpu").float().numpy())
        # qkv linear weight
        np.save(f"{CACHE_DIR}/megatron_qkv_linear_weight.npy", gpt_model.decoder.layers[0].self_attention.linear_qkv.weight.clone().detach().to("cpu").float().numpy())
        # linear proj weight
        np.save(f"{CACHE_DIR}/megatron_linear_proj_weight.npy", gpt_model.decoder.layers[0].self_attention.linear_proj.weight.clone().detach().to("cpu").float().numpy())
        # mlp linear fc1 weight
        np.save(f"{CACHE_DIR}/megatron_mlp_linear_fc1_weight.npy", gpt_model.decoder.layers[0].mlp.linear_fc1.weight.clone().detach().to("cpu").float().numpy())
        # mlp linear fc2 weight
        np.save(f"{CACHE_DIR}/megatron_mlp_linear_fc2_weight.npy", gpt_model.decoder.layers[0].mlp.linear_fc2.weight.clone().detach().to("cpu").float().numpy())
        # head weight
        np.save(f"{CACHE_DIR}/head_megatron.npy", gpt_model.output_layer.weight.clone().detach().to("cpu").float().numpy())
    if test_backward:
        output_megatron = gpt_model(input_megatron, position_ids_megatron, attention_mask_megatron)
        
        # Calculate loss and run backward
        labels_megatron = torch.randint(0, 184622, (1, SEQ_LEN), dtype=torch.int64, device="cuda")
        loss_megatron = F.cross_entropy(output_megatron.view(-1, 184622), labels_megatron.view(-1))
        loss_megatron.backward()
        
        # Store for comparison
        embed_grad_megatron = gpt_model.embedding.word_embeddings.weight.grad[0:10].clone().cpu()
        output_megatron = output_megatron.clone().detach().cpu()
    else:
        with torch.no_grad():
            output_megatron = gpt_model(input_megatron, position_ids_megatron, attention_mask_megatron).clone().detach().cpu()
    
    gpt_model.zero_grad()
    gpt_model.to("cpu")
    del gpt_model
    gc.collect()
    torch.cuda.empty_cache()
    destroy_distributed()

    print("Building hf model...")
    attention_mask_hf = torch.ones((1, SEQ_LEN), dtype=torch.float32, device="cuda").bool()
    # test hf model
    hf_model = AutoModelForCausalLM.from_pretrained(
        EMU3_HF_MODEL_DIR,
        device_map="cuda:0",
        torch_dtype=DTYPE_DICT[DTYPE],
        attn_implementation="eager",
        trust_remote_code=True,
    )
    print(hf_model)
    # check param dtype
    print(hf_model.model.layers[0].input_layernorm.weight.detach().cpu().float().numpy())
    hf_model.cuda()
    hf_model.eval()  # Switch to train mode
    if debug:
        for name, module in hf_model.named_modules():
            module.register_forward_hook(generate_hook(module, "hf_" + name))
        # rms norm weight
        np.save(f"{CACHE_DIR}/hf_rmsnorm_weight.npy", hf_model.model.layers[0].input_layernorm.weight.clone().detach().to("cpu").float().numpy())
        # q linear weight
        np.save(f"{CACHE_DIR}/hf_q_linear_weight.npy", hf_model.model.layers[0].self_attn.q_proj.weight.clone().detach().to("cpu").float().numpy())
        # k linear weight
        np.save(f"{CACHE_DIR}/hf_k_linear_weight.npy", hf_model.model.layers[0].self_attn.k_proj.weight.clone().detach().to("cpu").float().numpy())
        # v linear weight
        np.save(f"{CACHE_DIR}/hf_v_linear_weight.npy", hf_model.model.layers[0].self_attn.v_proj.weight.clone().detach().to("cpu").float().numpy())
        # o linear weight
        np.save(f"{CACHE_DIR}/hf_o_linear_weight.npy", hf_model.model.layers[0].self_attn.o_proj.weight.clone().detach().to("cpu").float().numpy())
        # gate linear weight
        np.save(f"{CACHE_DIR}/hf_gate_linear_weight.npy", hf_model.model.layers[0].mlp.gate_proj.weight.clone().detach().to("cpu").float().numpy())
        # up linear weight
        np.save(f"{CACHE_DIR}/hf_up_linear_weight.npy", hf_model.model.layers[0].mlp.up_proj.weight.clone().detach().to("cpu").float().numpy())
        # down linear weight
        np.save(f"{CACHE_DIR}/hf_down_linear_weight.npy", hf_model.model.layers[0].mlp.down_proj.weight.clone().detach().to("cpu").float().numpy())
        # head weight
        np.save(f"{CACHE_DIR}/head_hf.npy", hf_model.lm_head.weight.clone().detach().to("cpu").float().numpy())
    print("HF model built.")

    print("Running hf model forward and backward...")
    if test_backward:
        output_hf = hf_model(input_ids=embed_grad_hf, position_ids=position_ids_hf, attention_mask=attention_mask_hf).logits
        
        # Calculate loss and run backward with same labels
        labels_hf = labels_megatron.clone().detach()  # Use same labels as megatron
        loss_hf = F.cross_entropy(output_hf.view(-1, 184622), labels_hf.view(-1))
        loss_hf.backward()
        
        embed_grad_hf = hf_model.model.embed_tokens.weight.grad[0:10].clone().cpu()
        output_hf = output_hf.clone().detach().cpu()
    else:
        with torch.no_grad():
            output_hf = hf_model(input_ids=embed_grad_hf, position_ids=position_ids_hf, attention_mask=attention_mask_hf).logits.clone().detach().cpu()

    # Compare forward outputs
    assert check_tol(output_megatron, output_hf, atol=ATOL[DTYPE], rtol=RTOL[DTYPE]), "Output not close enough"
    if test_backward:
        assert check_tol(embed_grad_megatron, embed_grad_hf, atol=ATOL[DTYPE], rtol=RTOL[DTYPE]), "Input gradient not close enough"


if __name__ == "__main__":
    # torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cuda.matmul.allow_tf32 = True
    # run in block mode
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    test_model_forward_and_backward(debug=True, deterministic_mode=True, test_backward=False)