# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

'''Entry point for querying an index using a GPTChunkDataset.

Querying involves:

  - Iterate all chunks in the GPTChunkDataset.
  - Query index for neighbor chunk IDs (i.e., chunks from the chunk database).
  - Save neighbor chunk IDs to disk, for use in building a RetroDataset sample
      during pretraining.
'''

import numpy as np
import os
import psutil
import time
import torch
import typing
from tqdm import tqdm

from megatron.core.models.retro.data.config import RetroPreprocessingConfig
from megatron.core.models.retro.data.db.dataset import DBDataset
from megatron.core.models.retro.data.db.utils import (
    get_merged_train_dataset as get_db_merged_train_dataset,
)
from megatron.core.models.retro.data.external_libs import faiss, h5py
from megatron.core.models.retro.data.index.factory import IndexFactory
from megatron.core.models.retro.data.index.index import Index
from megatron.core.models.retro.data.index.utils import get_index_dir
from megatron.core.models.retro.data.query.chunk_dataset import GPTChunkDataset
from megatron.core.models.retro.data.utils import (
    GPTToTextDataset,
    get_blocks_by_rank,
    print_rank_0,
    retro_makedir,
)

from .gpt_chunk_dataset import build_gpt_chunk_datasets_from_gpt_datasets


def get_index(
    config: RetroPreprocessingConfig,
    ondisk: bool=False,
) -> faiss.Index:
    '''Read index from disk.'''

    # Load index.
    index_wrapper = IndexFactory.get_index(config.retro_index_type)
    index_dir = get_index_dir(config)
    added_index_path = index_wrapper.get_added_index_path(config)
    if ondisk:
        index = faiss.read_index(added_index_path, faiss.IO_FLAG_MMAP)
    else:
        index = faiss.read_index(added_index_path)

    # Search parameters.
    faiss.ParameterSpace().set_index_parameter(index, "efSearch", config.retro_query_ef_search)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", config.retro_query_nprobe)

    return index


def embed_block(
    config: RetroPreprocessingConfig,
    gpt_dataset: GPTChunkDataset,
    block: dict,
) -> np.ndarray:
    '''Embed block of chunks.'''
    text_block_dataset = torch.utils.data.Subset(
        GPTToTextDataset(gpt_dataset, config.retro_tokenizers.gpt), range(*block["range"]),
    )
    return config.retro_bert_embedders.mem.embed_text_dataset(text_block_dataset)


def query_embeddings(
    config: RetroPreprocessingConfig,
    db_dataset: DBDataset,
    index: Index,
    embeddings: np.ndarray,
    chunk_id_range: range,
    sample_map: dict,
    n_chunks_per_sample: int,
    verbose: bool=True,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    '''Query neighbors of a block of embeddings.

    Querying includes:
      - Query index for neighbor chunk IDs.
      - Filter chunk IDs that have the same document ID as the queried embedding.
    '''

    # Query neighbor ids.
    if verbose:
        print_rank_0("search.")
    t = time.time()
    assert index.ntotal > 0, "check we don't accidentally have an empty index."
    _, query_neighbor_ids = index.search(embeddings, config.retro_query_num_neighbors_query)
    if verbose:
        print_rank_0("  time : %.3f sec." % (time.time() - t))

    # Filter banned neighbor ids.
    if verbose:
        print_rank_0("filter banned neighbor ids.")
    filtered_neighbor_ids = np.full(
        shape=(len(query_neighbor_ids), config.retro_query_num_neighbors_save),
        fill_value=-1,
        dtype="int64",
    )
    min_chunk_id, max_chunk_id = chunk_id_range
    for chunk_id in range(min_chunk_id, max_chunk_id):

        sample_id = chunk_id // n_chunks_per_sample
        sample = sample_map[sample_id]
        sample_dataset_idx = sample["dataset_idx"].item()
        sample_doc_ids = sample["doc_ids"].tolist()
        sample_doc_tuples = [(sample_dataset_idx, d) for d in sample_doc_ids]

        # Get valid neighbors (!= -1).
        query_row = [i for i in query_neighbor_ids[chunk_id - min_chunk_id] if i >= 0]

        # Filter row.
        filtered_row = [
            i
            for i in query_row
            if tuple(db_dataset.doc_tuples[i].tolist()) not in sample_doc_tuples
        ]
        filtered_row = filtered_row[: config.retro_query_num_neighbors_save]
        filtered_row += [-1] * (config.retro_query_num_neighbors_save - len(filtered_row))
        filtered_neighbor_ids[chunk_id - min_chunk_id] = filtered_row

    return query_neighbor_ids, filtered_neighbor_ids


def query_embedding_block(
    config: RetroPreprocessingConfig,
    db_dataset: DBDataset,
    index: Index,
    embeddings: np.ndarray,
    chunk_id_range: range,
    sample_map: dict,
    n_chunks_per_sample: int,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    '''Query a block of embeddings.

    The block is broken into smaller sub-blocks, for easier tracking of progress.
    Both the raw neighbor IDs and the filtered neighbor IDs (i.e., chunks with the
    same document ID are removed) are collected.
    '''

    query_neighbor_ids = []
    filtered_neighbor_ids = []

    # Query in sub-blocks.
    partial_block_size = 1000
    for partial_start_idx in tqdm(
        range(0, len(embeddings), partial_block_size),
        "  search",
        miniters=(len(embeddings) // partial_block_size) // 10,
        disable=torch.distributed.get_rank() != 0,
    ):
        partial_end_idx = min(len(embeddings), partial_start_idx + partial_block_size)
        partial_embeddings = embeddings[partial_start_idx:partial_end_idx]
        partial_chunk_id_range = (
            chunk_id_range[0] + partial_start_idx,
            chunk_id_range[0] + partial_end_idx,
        )
        partial_query_neighbor_ids, partial_filtered_neighbor_ids = query_embeddings(
            config,
            db_dataset,
            index,
            partial_embeddings,
            partial_chunk_id_range,
            sample_map,
            n_chunks_per_sample,
            verbose=False,
        )
        query_neighbor_ids.append(partial_query_neighbor_ids)
        filtered_neighbor_ids.append(partial_filtered_neighbor_ids)

    # Concatenate.
    query_neighbor_ids = np.concatenate(query_neighbor_ids, axis=0)
    filtered_neighbor_ids = np.concatenate(filtered_neighbor_ids, axis=0)

    return query_neighbor_ids, filtered_neighbor_ids


def query_block_neighbors(
    config: RetroPreprocessingConfig,
    db_dataset: DBDataset,
    query_dataset: GPTChunkDataset,
    index: Index,
    block: dict,
) -> None:
    '''Query neighbors of a dataset block (i.e., range).'''

    n_chunks_per_sample = query_dataset.n_chunks_per_sample

    # Sample map.
    sample_ids = sorted(
        list(set(chunk_id // n_chunks_per_sample for chunk_id in range(*block["range"])))
    )
    sample_map = {}
    for i in sample_ids:
        sample = query_dataset.sample_dataset[i]
        sample_map[i] = {
            "dataset_idx": sample["dataset_id"],
            "doc_ids": sample["document_ids"],
        }

    # Embed block.
    embeddings = embed_block(config, query_dataset, block)

    # Query embeddings.
    _, filtered_neighbor_ids = query_embedding_block(
        config, db_dataset, index, embeddings, block["range"], sample_map, n_chunks_per_sample,
    )

    if config.retro_task_validate is None:
        # Save neighbors.
        print_rank_0("save neighbors.")
        retro_makedir(config, os.path.dirname(block["path"]))
        f = h5py.File(block["path"], "w")
        f.create_dataset("neighbors", data=filtered_neighbor_ids)
        f.close()

    else:
        # Validate neighbors.
        with h5py.File(block["path"]) as f:
            existing_neighbor_ids = np.copy(f["neighbors"])
            # >>>
            # import math
            # import os
            # n = (existing_neighbor_ids != filtered_neighbor_ids).sum() / existing_neighbor_ids.size
            # e = math.sqrt(((existing_neighbor_ids - filtered_neighbor_ids)**2).mean().item())
            # print("~~~~~~~~~~~ n %f, e %f ... '%s' ~~~~~~~~~~~" % (n, e, os.path.basename(block["path"])))

            # from lutil import pax
            # pax("existing_neighbor_ids, neighbor_ids")
            # <<<

            # >>>
            assert np.array_equal(existing_neighbor_ids, filtered_neighbor_ids)
            # <<<


def query_dataset_neighbors(
    config: RetroPreprocessingConfig,
    db_dataset: DBDataset,
    query_dataset: GPTChunkDataset,
    num_active_chunks: int,
    prefix: str,
    neighbor_dir: str,
    index: Index,
) -> None:
    '''Query neighbors of each chunk within a dataset.'''

    def validate(f: h5py.File) -> None:
        assert f["neighbors"].shape[1] == config.retro_query_num_neighbors_save, (
            "neighbors.shape == %s; num_neighbors_target == %d."
            % (str(f["neighbors"].shape), config.retro_num_neighbors_target,)
        )

    if config.retro_task_validate is None:
        retro_makedir(config, neighbor_dir)
        blocks = get_blocks_by_rank(
            neighbor_dir, num_active_chunks, config.retro_block_size, validate=validate,
        )
        active_blocks = blocks.missing
    else:
        blocks = get_blocks_by_rank(
            neighbor_dir,
            num_active_chunks,
            config.retro_block_size,
            validate=validate,
            sample=config.retro_task_validate,
        )
        assert blocks.n_missing_world == 0
        active_blocks = blocks.existing

    # Query each block.
    for block_index, block in enumerate(active_blocks):

        if block is not None:

            # Progress.
            print_rank_0(
                "%squery '%s' block %d / %d ... %s ... mem %.3f gb, %.1f%%."
                % (
                    "" if config.retro_task_validate is None else "[validate] ",
                    prefix,
                    block_index,
                    len(active_blocks),
                    os.path.basename(block["path"]),
                    psutil.virtual_memory()[3] / 1024 ** 3,
                    psutil.virtual_memory()[2],
                )
            )

            # Query block neighbors.
            query_block_neighbors(config, db_dataset, query_dataset, index, block)

        # Synchronize progress across all ranks. (for easier observation)
        print_rank_0(" > waiting for other ranks to finish block.")
        torch.distributed.barrier()


def query_neighbors(config: RetroPreprocessingConfig) -> None:
    '''Query pretraining datasets (train & valid).'''

    # Num threads.
    faiss.omp_set_num_threads(64)

    # Load chunk db dataset.
    print_rank_0("load chunk db dataset.")
    db_dataset = get_db_merged_train_dataset(
        project_dir=config.retro_project_dir,
        chunk_length=config.retro_gpt_chunk_length,
        eod_token_id=config.retro_tokenizers.gpt.eod,
    )
    db_dataset.load_doc_tuples()

    # Load index.
    print_rank_0(" > get index.")
    index = get_index(config)

    # Query each (i.e., train, valid, test) dataset.
    print_rank_0(" > query.")
    for prefix, info in vars(config.retro_gpt_chunk_datasets).items():
        if info is None:
            continue
        print_rank_0(" > query '%s' dataset ... %d samples." % (prefix, info["num_active_chunks"]))
        query_dataset_neighbors(
            config,
            db_dataset,
            info["dataset"],
            info["num_active_chunks"],
            prefix,
            info["neighbor_dir"],
            index,
        )