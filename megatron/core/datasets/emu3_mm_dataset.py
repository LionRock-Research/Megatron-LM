# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import numpy
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from megatron.core.datasets.utils_s3 import S3Config, is_s3_path
from megatron.core.datasets.gpt_dataset import _build_document_index, _build_shuffle_index
from megatron.core.utils import log_single_rank
logger = logging.getLogger(__name__)


@dataclass
class EMU3MMDatasetConfig(GPTDatasetConfig):
    """Configuration object for EMU3MM datasets that extends GPTDatasetConfig"""

    token_num_per_visual_block: int = 4096
    """Optional list of token IDs to add as prefix before each document segment"""
    def __post_init__(self):
        assert self.sequence_length > self.token_num_per_visual_block, f"sequence_length ({self.sequence_length}) should be greater than token_num_per_visual_block ({self.token_num_per_visual_block})"

class EMU3MMDataset(GPTDataset):
    """EMU3MM dataset that extends GPTDataset to support document prefixes

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the EMU3MMDataset
        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping
        indexed_indices (numpy.ndarray): The set of the documents indices to expose
        num_samples (Optional[int]): The number of samples to draw from the indexed dataset
        index_split (Split): The indexed_indices Split
        config (EMU3MMDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: EMU3MMDatasetConfig,
    ) -> None:
        if config.use_prefix:
            self.prefix_dataset = indexed_dataset[1]
        super().__init__(indexed_dataset[0], dataset_path, indexed_indices, num_samples, index_split, config)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: EMU3MMDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        # remove '_document'
        prefix_path = dataset_path[:-9] + '_prefix_document'
        try:
            if is_s3_path(prefix_path):
                prefix_low_level_dataset = IndexedDataset(
                    prefix_path,
                    multimodal=False,
                    mmap=config.mmap_bin_files,
                    s3_config=S3Config(path_to_idx_cache=config.s3_cache_path),
                )
            else:
                prefix_low_level_dataset = IndexedDataset(
                    prefix_path,
                    multimodal=False,
                    mmap=config.mmap_bin_files,
                )
        except Exception as e:
            print(f"Error building prefix dataset: {e}")
            raise e
        if is_s3_path(dataset_path):
            low_level_dataset = IndexedDataset(
                dataset_path,
                multimodal=False,
                mmap=config.mmap_bin_files,
                s3_config=S3Config(path_to_idx_cache=config.s3_cache_path),
            )
        else:
            low_level_dataset = IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)
        return low_level_dataset, prefix_low_level_dataset

    def _query_document_sample_shuffle_indices(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # TODO: merge this two conditions
        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])
            # Add the entire sample
            sample_part = self.dataset.get(
                self.document_index[doc_index_beg],
                offset=doc_index_beg_offset,
                length=doc_index_end_offset - doc_index_beg_offset + self.config.add_extra_token_to_sequence,
            )
            prefix = self.prefix_dataset.get(self.document_index[doc_index_beg])
            sample_part = numpy.concatenate([prefix, sample_part], axis=0)

            sample_parts.append(sample_part)

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + self.config.add_extra_token_to_sequence
                sample_part = self.dataset.get(self.document_index[i], offset=offset, length=length)
                prefix = self.prefix_dataset.get(self.document_index[i])
                sample_part = numpy.concatenate([prefix, sample_part], axis=0)
                sample_parts.append(sample_part)
        
        assert len(document_ids) == len(
            sample_parts
        ), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

        length = sum(map(len, sample_parts))

        # Pad the sample if necessary
        if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
            sample_parts.append(
                [self._pad_token_id] * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
            )

        return (
            numpy.concatenate(sample_parts, dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )
    
    def _get_tokens_per_epoch(self) -> int:
        """Get the number of tokens per epoch

        Returns:
            int: The number of tokens per epoch
        """
        return int(numpy.sum(self.dataset.sequence_lengths[self.indices])) + int(numpy.sum(self.prefix_dataset.sequence_lengths[self.indices]))
            
    def _build_document_sample_shuffle_indices_helper(
        self, path_to_cache, path_to_description, path_to_document_index, path_to_sample_index, path_to_shuffle_index
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build and save the document index, sample index, and shuffle index

        Args:
            path_to_cache (str): The path to the cache

            path_to_description (str): The path to the description

            path_to_document_index (str): The path to the document index

            path_to_sample_index (str): The path to the sample index

            path_to_shuffle_index (str): The path to the shuffle index
        """
        t_beg = time.time()
        
        separate_final_epoch = False
        numpy_random_state = numpy.random.RandomState(self.config.random_seed)
        num_total_videos = len(self.indices)
        token_nums_per_sample = self.config.sequence_length + self.config.add_extra_token_to_sequence
        requested_num_samples = self.num_samples if self.num_samples is not None else float('-inf')
        
        sample_index = []
        current_epoch_ducument_indexes = _build_document_index(self.indices, 1, numpy_random_state, separate_final_epoch=False)
        document_index = [current_epoch_ducument_indexes]
        
    
        # FIXME: probably need to control the second nums
        # FIXME: separate_final_epoch=False
        
        num_samples = 0
        idx = 0
        num_epochs = 0
        remained_unpaced_token_nums = 0
        
        while num_samples < requested_num_samples and num_epochs < 1:
            video_idx = current_epoch_ducument_indexes[idx % num_total_videos]
            visual_token_length = self.dataset.sequence_lengths[video_idx]
            prefix_length = self.prefix_dataset.sequence_lengths[video_idx]
            num_visual_blocks = (visual_token_length + prefix_length) // self.config.token_num_per_visual_block
            assert num_visual_blocks * self.config.token_num_per_visual_block == visual_token_length, f"visual_token_length ({visual_token_length}) is not a multiple of token_num_per_visual_block ({self.config.token_num_per_visual_block})"
            packed_visual_blocks = 0
            
            while packed_visual_blocks < num_visual_blocks:
                if remained_unpaced_token_nums > 0:
                    consumed_visual_blocks = (token_nums_per_sample - prefix_length) // self.config.token_num_per_visual_block
                else:
                    consumed_visual_blocks = num_visual_blocks
                # packed to previous sample if remained space is enough
                if consumed_visual_blocks > num_visual_blocks - packed_visual_blocks:
                    # add start
                    sample_index.append([idx, packed_visual_blocks * self.config.token_num_per_visual_block])
                    remained_unpaced_token_nums = token_nums_per_sample - prefix_length - num_visual_blocks * self.config.token_num_per_visual_block
                    # move to next sample
                    packed_visual_blocks = num_visual_blocks
                else:
                    packed_visual_blocks += consumed_visual_blocks
                    # add end
                    sample_index.append([idx, packed_visual_blocks * self.config.token_num_per_visual_block + 1])
                    
                    num_samples += 1
                    remained_unpaced_token_nums = 0
            
            idx += 1
            if idx >= num_total_videos * num_epochs:
                num_epochs += 1
                current_epoch_ducument_indexes = _build_document_index(self.indices, num_epochs, numpy_random_state, separate_final_epoch=False)
                document_index.append(current_epoch_ducument_indexes)
        
        if idx > num_total_videos * num_epochs:
            num_epochs += 1
            num_samples_from_final_epoch = idx // num_total_videos
            num_samples_sans_final_epoch = num_samples - num_samples_from_final_epoch
            document_index[-1] = document_index[-1][:num_samples_from_final_epoch]
            threshold = 0.80
            if num_samples_from_final_epoch < int(threshold * num_samples_sans_final_epoch // (num_epochs - 1)): 
                separate_final_epoch = True
            log_single_rank(logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}")
        else:
            num_samples_sans_final_epoch = num_samples
        
        batch_document_index = numpy.concatenate(document_index)
        sample_index = numpy.concatenate(sample_index)
        shuffle_index = _build_shuffle_index(num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state)

            
        if path_to_cache:
            os.makedirs(path_to_cache, exist_ok=True)
            with open(path_to_description, "wt") as writer:
                writer.write(self.unique_description)
            numpy.save(path_to_document_index, batch_document_index, allow_pickle=True)
            numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
            numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
        else:
            log_single_rank(logger, logging.WARNING, f"Unable to save the {type(self).__name__} indexes because path_to_cache is None")
        
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}")
        log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")
        return document_index, sample_index, shuffle_index