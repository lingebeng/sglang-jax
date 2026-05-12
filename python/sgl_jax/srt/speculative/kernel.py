"""Pure JAX/NumPy implementation of EAGLE tree structure building.

Replaces the Pallas kernel in build_eagle_tree_structure_kernel.py with
a portable implementation that works on any JAX backend (CPU/GPU/TPU).
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp


def build_eagle_tree_structure_jax(
    parent_list: np.ndarray | jnp.ndarray,
    selected_index: np.ndarray | jnp.ndarray,
    verified_seq_len: np.ndarray | jnp.ndarray,
    draft_token_num: int,
    topk: int,
    seq_lens_sum: int,
    max_context_len: int,
    tree_mask_mode: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build EAGLE tree structure in pure JAX/NumPy.

    Args:
        parent_list: Parent indices, shape (bs, topk*(depth-1)+1).
        selected_index: Selected token indices from top-k, shape (bs, draft_token_num-1).
        verified_seq_len: Sequence lengths per request, shape (bs,).
        draft_token_num: Number of draft tokens (= num_verify_tokens).
        topk: Top-k value used in draft.
        seq_lens_sum: Sum of all sequence lengths.
        max_context_len: Max context length per request.
        tree_mask_mode: 0=FULL_MASK, 1=COMPACT (draft-only mask, expanded before return).

    Returns:
        (tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling)
    """
    parent_list = np.asarray(parent_list, dtype=np.int32)
    selected_index = np.asarray(selected_index, dtype=np.int32)
    verified_seq_len = np.asarray(verified_seq_len, dtype=np.int32)
    bs = parent_list.shape[0]

    # --- Step A: Build compact tree mask (bs, draft_token_num, draft_token_num) ---
    # compact_mask[bid, tid, j] = 1 means token tid can see token j
    compact_mask = np.zeros((bs, draft_token_num, draft_token_num), dtype=np.int32)

    # --- Step B: Compute positions ---
    positions = np.zeros(bs * draft_token_num, dtype=np.int32)

    # --- Step C: Build retrive structures ---
    retrive_index = np.full((bs, draft_token_num), -1, dtype=np.int32)
    retrive_next_token = np.full((bs, draft_token_num), -1, dtype=np.int32)
    retrive_next_sibling = np.full((bs, draft_token_num), -1, dtype=np.int32)

    for bid in range(bs):
        seq_len = int(verified_seq_len[bid])
        sel_idx = selected_index[bid]  # (draft_token_num - 1,)
        parents = parent_list[bid]

        # --- Process each token ---
        for tid in range(draft_token_num):
            global_token_idx = bid * draft_token_num + tid

            if tid == 0:
                # Verified token (root) can see itself
                compact_mask[bid, 0, 0] = 1
                # Verified token
                positions[global_token_idx] = seq_len
                retrive_index[bid, tid] = global_token_idx
            else:
                # Draft token: trace back to root via parent chain
                cur = tid - 1  # 0-indexed into selected_index
                depth = 0

                while True:
                    depth += 1
                    # Mark ancestor visible
                    compact_mask[bid, tid, cur + 1] = 1  # cur+1 because tid=0 is verified

                    parent_tb_idx = int(sel_idx[cur]) // topk
                    if parent_tb_idx == 0:
                        # Reached root (verified token)
                        compact_mask[bid, tid, 0] = 1
                        break

                    # Find parent's position in selected_index
                    parent_token_idx = int(parents[parent_tb_idx])
                    found = False
                    for j in range(draft_token_num - 1):
                        if int(sel_idx[j]) == parent_token_idx:
                            cur = j
                            found = True
                            break
                    if not found:
                        # Parent not found in selected set, link to root
                        compact_mask[bid, tid, 0] = 1
                        break

                positions[global_token_idx] = seq_len + depth
                retrive_index[bid, tid] = global_token_idx

        # --- Build retrive_next_token / retrive_next_sibling ---
        # Process tokens from last to first (backwards), matching the Pallas kernel
        for i in range(draft_token_num - 1, 0, -1):
            retrive_index[bid, i] = bid * draft_token_num + i

            parent_tb_idx = int(sel_idx[i - 1]) // topk

            if parent_tb_idx > 0:
                # Find parent position in selected_index
                parent_token_idx = int(parents[parent_tb_idx])
                parent_position = 0
                for j in range(draft_token_num - 1):
                    if int(sel_idx[j]) == parent_token_idx:
                        parent_position = j + 1  # +1 because position 0 is verified token
                        break
            else:
                parent_position = 0

            if parent_position < draft_token_num:
                if retrive_next_token[bid, parent_position] == -1:
                    retrive_next_token[bid, parent_position] = i
                else:
                    origin_next = retrive_next_token[bid, parent_position]
                    retrive_next_token[bid, parent_position] = i
                    retrive_next_sibling[bid, i] = origin_next

        retrive_index[bid, 0] = bid * draft_token_num

    # --- Step D: Build full mask ---
    tree_mask = _expand_compact_to_full_mask(
        compact_mask, verified_seq_len, draft_token_num, seq_lens_sum, max_context_len, bs
    )

    return (
        jnp.asarray(tree_mask, dtype=jnp.int32),
        jnp.asarray(positions, dtype=jnp.int32),
        jnp.asarray(retrive_index, dtype=jnp.int32),
        jnp.asarray(retrive_next_token, dtype=jnp.int32),
        jnp.asarray(retrive_next_sibling, dtype=jnp.int32),
    )


def _expand_compact_to_full_mask(
    compact_mask: np.ndarray,
    verified_seq_len: np.ndarray,
    draft_token_num: int,
    seq_lens_sum: int,
    max_context_len: int,
    bs: int,
) -> np.ndarray:
    """Expand compact mask (bs, draft_token_num, draft_token_num) to FULL_MASK format.

    FULL_MASK layout (flattened 1D):
      For each batch bid, for each token tid:
        row = [1]*seq_len + compact_mask[bid, tid, :]
        (length = seq_len + draft_token_num)
      All rows from all batches are concatenated.

    Returns:
        1D int32 array of the full mask.
    """
    # Total size = sum over all batches of (seq_len + draft_token_num) * draft_token_num
    total_size = seq_lens_sum * draft_token_num + draft_token_num * draft_token_num * bs
    # Capacity (padded for static shape)
    capacity = max_context_len * draft_token_num * bs + draft_token_num * draft_token_num * bs

    full_mask = np.ones(capacity, dtype=np.int32)
    # Zero out unused region
    if total_size < capacity:
        full_mask[total_size:] = 0

    offset = 0
    for bid in range(bs):
        seq_len = int(verified_seq_len[bid])
        row_len = seq_len + draft_token_num
        for tid in range(draft_token_num):
            # Prefix part: all visible (already 1)
            # Tree part: copy from compact_mask
            start = offset + seq_len
            full_mask[start: start + draft_token_num] = compact_mask[bid, tid]
            offset += row_len

    return full_mask
