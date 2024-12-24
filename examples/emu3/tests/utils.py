import os
import torch

from megatron.core import parallel_state

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