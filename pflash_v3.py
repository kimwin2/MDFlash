import heapq
import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from dflash import dflash_generate, cuda_time, empty_stage_times
from ddtree import (
    DDTREE_TREE_BUILD_STAGE_ORDER,
    compile_ddtree_tree,
    follow_verified_tree,
    compact_dynamic_cache,
)
from pexpress import build_perturbed_noise_embedding_batch


PFLASH_V3_BATCH_SIZE = 4
PFLASH_V3_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")


def normalize_branch_top_log_probs(
    top_log_probs: torch.Tensor,
) -> torch.Tensor:
    branch_means = top_log_probs.mean(dim=-1, keepdim=True)
    branch_stds = top_log_probs.std(dim=-1, keepdim=True, unbiased=False).clamp_min_(1e-6)
    return (top_log_probs - branch_means) / branch_stds


def build_round_robin_pflash_v3_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
    dict[str, float],
    list[int],
]:
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    if budget <= 0 or draft_logits.shape[0] == 0 or draft_logits.shape[1] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
            [],
        )

    num_branches, depth_limit, vocab_size = draft_logits.shape
    topk = min(max(budget, 1), vocab_size)

    copy_start = cuda_time()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    normalized_top_log_probs_cpu = normalize_branch_top_log_probs(top_log_probs_cpu)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = normalized_top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    heap_start = time.perf_counter()
    branch_queues: list[list[tuple[float, tuple[int, ...], int, int, float]]] = [[] for _ in range(num_branches)]
    for branch_idx in range(num_branches):
        first_logw = float(top_log_probs_np[branch_idx, 0, 0])
        heapq.heappush(branch_queues[branch_idx], (-first_logw, (0,), 1, 0, first_logw))

    parents = [-1]
    child_maps: list[dict[int, int]] = [dict()]
    node_token_ids: list[int] = []
    node_depths: list[int] = []
    node_source_branches: list[int] = []
    path_to_index: dict[tuple[int, ...], int] = {(): 0}

    branch_idx = 0
    skipped_branches = 0
    while len(node_token_ids) < budget and skipped_branches < num_branches:
        queue = branch_queues[branch_idx]
        added_node = False

        while queue and len(node_token_ids) < budget:
            _, ranks, depth, rank, logw = heapq.heappop(queue)
            prefix_tokens = tuple(
                int(top_token_ids_np[branch_idx, prefix_depth, prefix_rank])
                for prefix_depth, prefix_rank in enumerate(ranks)
            )

            if rank + 1 < topk:
                sibling_ranks = ranks[:-1] + (rank + 1,)
                sibling_logw = logw - float(top_log_probs_np[branch_idx, depth - 1, rank]) + float(
                    top_log_probs_np[branch_idx, depth - 1, rank + 1]
                )
                heapq.heappush(
                    queue,
                    (-sibling_logw, sibling_ranks, depth, rank + 1, sibling_logw),
                )

            if depth < depth_limit:
                child_ranks = ranks + (0,)
                child_logw = logw + float(top_log_probs_np[branch_idx, depth, 0])
                heapq.heappush(
                    queue,
                    (-child_logw, child_ranks, depth + 1, 0, child_logw),
                )

            if prefix_tokens in path_to_index:
                continue

            parent_path = prefix_tokens[:-1]
            parent_index = path_to_index.get(parent_path)
            if parent_index is None:
                raise RuntimeError("P-Flash V3 encountered a missing parent path during round-robin tree assembly.")

            token_id = prefix_tokens[-1]
            current_index = len(parents)
            parents.append(parent_index)
            child_maps.append(dict())
            child_maps[parent_index][token_id] = current_index
            path_to_index[prefix_tokens] = current_index
            node_token_ids.append(token_id)
            node_depths.append(depth)
            node_source_branches.append(branch_idx)
            added_node = True
            break

        if added_node:
            skipped_branches = 0
        else:
            skipped_branches += 1

        branch_idx = (branch_idx + 1) % num_branches

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + len(node_token_ids)
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = parents[index]
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    return (
        torch.tensor(node_token_ids, dtype=torch.long),
        torch.tensor(node_depths, dtype=torch.long),
        parents,
        child_maps,
        torch.from_numpy(visibility_np),
        build_subtimes,
        node_source_branches,
    )


@torch.inference_mode()
def pflash_v3_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    perturbation_temperature: float = 0.75,
    position_temperature_decay: float = 0.0,
    save_tree_traces: bool = False,
) -> SimpleNamespace:
    if block_size <= 1:
        return dflash_generate(
            model=model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    tree_budget = draft_horizon if tree_budget is None else max(tree_budget, 0)
    max_tree_nodes = 1 + tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(PFLASH_V3_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = cuda_time()
        base_noise_embedding = target.model.embed_tokens(block_output_ids)
        noise_embedding_batch = build_perturbed_noise_embedding_batch(
            base_noise_embedding=base_noise_embedding,
            num_branches=PFLASH_V3_BATCH_SIZE,
            perturbation_temperature=perturbation_temperature,
            position_temperature_decay=position_temperature_decay,
        )
        projected_target_hidden = model.project_target_hidden(target_hidden)
        batched_target_hidden = projected_target_hidden.expand(PFLASH_V3_BATCH_SIZE, -1, -1)
        draft_position_ids = position_ids[
            :,
            past_key_values_draft.get_seq_length() : start + block_size,
        ].expand(PFLASH_V3_BATCH_SIZE, -1)
        draft_logits = target.lm_head(model(
            target_hidden=batched_target_hidden,
            target_hidden_is_projected=True,
            noise_embedding=noise_embedding_batch,
            position_ids=draft_position_ids,
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        (
            node_token_ids,
            node_depths,
            parents,
            child_maps,
            visibility_cpu,
            tree_build_subtimes,
            node_source_branches,
        ) = build_round_robin_pflash_v3_tree(
            draft_logits=draft_logits,
            budget=tree_budget,
        )
        stage_times["tree_build"] += cuda_time() - tree_build_start
        for stage_name, stage_elapsed in tree_build_subtimes.items():
            stage_times[stage_name] += stage_elapsed

        tree_compile_start = cuda_time()
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start, previous_tree_length = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "accepted_indices": [int(index) for index in accepted_indices],
                "tree": {
                    "node_token_ids": [int(token_id) for token_id in node_token_ids.tolist()],
                    "node_depths": [int(depth) for depth in node_depths.tolist()],
                    "node_source_branches": [int(branch_idx) for branch_idx in node_source_branches],
                    "parents": [int(parent) for parent in parents],
                },
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_trees=round_trees,
    )
