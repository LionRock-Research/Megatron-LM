import torch
import os
import numpy as np

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.training.checkpointing import _load_base_checkpoint
from modelopt.torch.opt.plugins import restore_modelopt_state
from types import SimpleNamespace
from transformers import AutoModelForCausalLM

EMU3_HF_MODEL_DIR = "/data/models/Emu3-Gen"
EMU3_MEGATRON_MODEL_DIR = "/data/models/Emu3-Gen-Megatron"
SEQ_LEN = 128

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
    args = SimpleNamespace(load=EMU3_MEGATRON_MODEL_DIR, non_persistent_global_ckpt_dir=None, exit_on_missing_checkpoint=True, non_persistent_ckpt_type='global')
    state_dict, _, _, _ = _load_base_checkpoint(
        load_dir=args.load,
        args=args,
        rank0=False,
        checkpointing_context=None,
    )
    restore_modelopt_state([gpt_model], state_dict)
    gpt_model.cuda()
    torch.cuda.empty_cache()


def test_model_forward_and_backward():

    # prepare input
    input_hf = torch.randint(0, 184622, (1, SEQ_LEN), dtype=torch.int32, device="cuda")
    input_megatron = input_hf.detach().clone()
    position_ids_hf = torch.arange(SEQ_LEN, dtype=torch.int32, device="cuda").unsqueeze(0)
    position_ids_megatron = position_ids_hf.detach().clone()
    attention_mask_hf = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), dtype=torch.bool, device="cuda")).unsqueeze(0).unsqueeze(0)
    attention_mask_megatron = attention_mask_hf.detach().clone()

    # test megatron model first
    initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(42)
    # build megatron model
    print("Building megatron model...")
    transformer_config = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        attention_softmax_in_fp32=True,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_func=F.swi
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=184622,
        max_sequence_length=SEQ_LEN,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rope_scaling=None,
        rotary_base=1000000,
    )
    gpt_model.eval()
    print(gpt_model)
    load_weights(gpt_model)
    print("Megatron model built.")

    print("Running megatron model forward...")
    output_megatron = gpt_model(input_megatron, position_ids_megatron, attention_mask_megatron).detach().to("cpu").float().numpy()
    print(output_megatron.shape)
    del gpt_model
    torch.cuda.empty_cache()
    
    print("Building hf model...")
    # build hf model
    hf_model = AutoModelForCausalLM.from_pretrained(
        EMU3_HF_MODEL_DIR,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    hf_model.cuda()
    hf_model.eval()
    print("HF model built.")
    
    print("Running hf model forward...")
    output_hf = hf_model(input_ids=input_hf, position_ids=position_ids_hf).logits.detach().to("cpu").float().numpy()
    np.save("output_hf.npy", output_hf)
    np.save("output_megatron.npy", output_megatron)

    assert np.allclose(output_megatron, output_hf), f"forward fails with diff {np.abs(output_megatron-output_hf).mean()}"


if __name__ == "__main__":
    test_model_forward_and_backward()
