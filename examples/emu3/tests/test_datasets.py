
import sys
import torch

from typing import List, Optional, Tuple
from megatron.training.utils import (
    get_blend_and_blend_per_split,
)
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
# from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, GPTDataset
from megatron.core.datasets.emu3_mm_dataset import EMU3MMDataset, EMU3MMDatasetConfig
from megatron.training.arguments import parse_args
from megatron.training.tokenizer import build_tokenizer
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler, MegatronPretrainingRandomSampler
from utils import initialize_distributed, destroy_distributed
from megatron.core import mpu

def build_pretraining_data_loader(dataset, args, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       )

def core_gpt_dataset_config_from_args(args, tokenizer):

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)
    return EMU3MMDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        renormalize_blend_weights=args.renormalize_blend_weights,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        eod_mask_loss=args.eod_mask_loss,
        s3_cache_path=args.s3_cache_path,
        reset_position_ids=True,
        reset_attention_mask=True,
        create_attention_mask=True,
        token_num_per_visual_block=8,
    )

if __name__ == '__main__':
    sys.argv = [
        'script.py',
        '--seed', '42',
        '--split', '33,33,33',
        #'--data-path', '/datasets/MegatronTest/EMU3_TestDATA',
        '--data-args-path', '/data/datasets/MegatronTest/data_args.txt',
        '--tokenizer-type', 'HuggingFaceTokenizer',
        '--tokenizer-model', '/data/models/Emu3-Stage1',
        '--seq-length', '128',
        "--dataloader-type", "single",
        "--micro-batch-size", "4",
    ]
    initialize_distributed()
    args = parse_args()
    tokenizer = build_tokenizer(args)
    config = core_gpt_dataset_config_from_args(args, tokenizer=tokenizer)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        EMU3MMDataset,
        [400,400,400],
        lambda: True,
        config
    ).build()
    dataloader = iter(build_pretraining_data_loader(train_ds, args, 0))
    for _ in range(2):
        data = next(dataloader)
        print(data['labels'].shape)
    destroy_distributed()
