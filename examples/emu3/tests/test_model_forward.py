import torch
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

def check_tol(t1, t2, atol=5e-2, rtol=1e-8):
    # atol from https://github.com/NVIDIA/TransformerEngine/blob/v1.2.1/tests/pytorch/test_numerics.py#L888
    result = torch.allclose(t1, t2, atol=atol, rtol=rtol)
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

def generate_hook(module, suffix):
    def debug_hook(module, input, output):
        output_file_name = f"{CACHE_DIR}/{suffix}.npy"
        if isinstance(output, tuple):
            for i, o in enumerate(output):
                try:
                    np.save(f"{output_file_name}_o{i}.npy", o.detach().to("cpu").float().numpy())
                except Exception as e:
                    print(f"error saving {suffix}'s output {i}: {e}")
        else:
            try:
                np.save(output_file_name, output.detach().to("cpu").float().numpy())
            except Exception as e:
                print(f"error saving {suffix}'s output: {e}")

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


def set_deterministic_mode():
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_model_forward_and_backward(debug=False):
    # set deterministic mode
    set_deterministic_mode()

    initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(SEED)
    # prepare input
    input_hf = torch.randint(0, 184622, (1, SEQ_LEN), dtype=torch.int32, device="cuda")
    input_megatron = input_hf.detach().clone()
    position_ids_hf = torch.arange(SEQ_LEN, dtype=torch.int32, device="cuda").unsqueeze(0)
    position_ids_megatron = position_ids_hf.detach().clone()
    attention_mask_megatron = torch.ones((1, 1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.bool, device="cuda")

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
    #    bf16=True,
        ffn_hidden_size=14336,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=True,
        add_bias_linear=False,
        deterministic_mode=True,
    )
    print(transformer_config)
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=184622,
        max_sequence_length=SEQ_LEN,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rope_scaling=False,
        rotary_base=1000000.0,
        pre_process=True,
        post_process=True,
    )
    print(gpt_model)
    load_weights(gpt_model)
    print("Megatron model built.")

    print("Running megatron model forward...")
    gpt_model.eval()
    if debug:
        for name, module in gpt_model.named_modules():
            module.register_forward_hook(generate_hook(module, "megatron_" + name))
    with torch.no_grad():
        output_megatron = (
            gpt_model(input_megatron, attention_mask_megatron, position_ids_megatron).detach()
        )
    print(output_megatron.shape)
    del gpt_model
    torch.cuda.empty_cache()
    

    print("Building hf model...")
    # test hf model
    hf_model = AutoModelForCausalLM.from_pretrained(
        EMU3_HF_MODEL_DIR,
        device_map="cuda:0",
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    print(hf_model)
    hf_model.cuda()
    hf_model.eval()
    if debug:
        for name, module in hf_model.named_modules():
            module.register_forward_hook(generate_hook(module, "hf_" + name))
    print("HF model built.")

    print("Running hf model forward...")
    with torch.no_grad():
        output_hf = hf_model(input_ids=input_hf, position_ids=position_ids_hf).logits.detach()
    print(output_hf.shape)
    np.save("output_hf.npy", output_hf.cpu().float().numpy())
    np.save("output_megatron.npy", output_megatron.cpu().float().numpy())

    assert check_tol(output_megatron, output_hf)


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    test_model_forward_and_backward(debug=True)