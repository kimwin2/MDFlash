import copy
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from dflash import dflash_generate, cuda_time, empty_stage_times
from ddtree import (
    DDTREE_TREE_BUILD_STAGE_ORDER,
    build_ddtree_tree,
    compact_dynamic_cache,
)
from pexpress import build_perturbed_noise_embedding_batch


PFLASH_V2_BATCH_SIZE = 4
PFLASH_V2_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")


def _repeat_cache_batch(cache_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if cache_tensor.numel() == 0:
        return cache_tensor
    if cache_tensor.shape[0] != 1:
        raise ValueError("P-Flash V2 expects the verifier cache to have batch size 1 before expansion.")
    return cache_tensor.expand(batch_size, *cache_tensor.shape[1:]).clone()


def repeat_dynamic_cache_batch(
    past_key_values: DynamicCache,
    batch_size: int,
) -> DynamicCache:
    expanded_cache = copy.deepcopy(past_key_values)

    if hasattr(expanded_cache, "key_cache") and hasattr(expanded_cache, "value_cache"):
        expanded_cache.key_cache = [
            _repeat_cache_batch(key_cache, batch_size)
            for key_cache in expanded_cache.key_cache
        ]
        expanded_cache.value_cache = [
            _repeat_cache_batch(value_cache, batch_size)
            for value_cache in expanded_cache.value_cache
        ]
        return expanded_cache

    if hasattr(expanded_cache, "layers"):
        for layer in expanded_cache.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            layer.keys = _repeat_cache_batch(layer.keys, batch_size)
            layer.values = _repeat_cache_batch(layer.values, batch_size)
        return expanded_cache

    raise RuntimeError("Unsupported DynamicCache layout for P-Flash V2 batch expansion.")


def select_dynamic_cache_batch(
    past_key_values: DynamicCache,
    batch_index: int,
) -> DynamicCache:
    selected_cache = copy.deepcopy(past_key_values)

    if hasattr(selected_cache, "key_cache") and hasattr(selected_cache, "value_cache"):
        selected_cache.key_cache = [
            key_cache[batch_index : batch_index + 1].contiguous()
            for key_cache in selected_cache.key_cache
        ]
        selected_cache.value_cache = [
            value_cache[batch_index : batch_index + 1].contiguous()
            for value_cache in selected_cache.value_cache
        ]
        return selected_cache

    if hasattr(selected_cache, "layers"):
        for layer in selected_cache.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            layer.keys = layer.keys[batch_index : batch_index + 1].contiguous()
            layer.values = layer.values[batch_index : batch_index + 1].contiguous()
        return selected_cache

    raise RuntimeError("Unsupported DynamicCache layout for P-Flash V2 batch selection.")


def compile_batched_tree_group(
    root_token_id: int,
    start: int,
    trees: list[SimpleNamespace],
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], int]:
    max_tree_nodes = int(verify_input_ids_buffer.shape[1])

    if previous_tree_start is not None:
        attention_mask_buffer[:, 0, :, previous_tree_start : previous_tree_start + max_tree_nodes] = 0

    verify_input_ids_buffer.fill_(root_token_id)
    verify_position_ids_buffer.fill_(start)
    tree_visibility_buffer.zero_()

    diag_indices = torch.arange(max_tree_nodes, device=device)
    tree_visibility_buffer[:, diag_indices, diag_indices] = True

    current_lengths = []
    for batch_idx, tree in enumerate(trees):
        current_length = 1 + int(tree.node_token_ids.numel())
        current_lengths.append(current_length)

        if current_length > 1:
            verify_input_ids_buffer[batch_idx, 1:current_length].copy_(tree.node_token_ids, non_blocking=False)
            verify_position_ids_buffer[batch_idx, 1:current_length].copy_(tree.node_depths, non_blocking=False)
            verify_position_ids_buffer[batch_idx, 1:current_length].add_(start)

        tree_visibility_buffer[batch_idx, :current_length, :current_length].copy_(tree.visibility_cpu, non_blocking=False)

    tree_block = attention_mask_buffer[:, 0, :, start : start + max_tree_nodes]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(tree_visibility_buffer, 0)

    attention_mask = attention_mask_buffer[:, :, :, : start + max_tree_nodes]
    return verify_input_ids_buffer, verify_position_ids_buffer, attention_mask, current_lengths, start


def follow_verified_forest(
    child_maps_batch: list[list[dict[int, int]]],
    posterior_logits: torch.Tensor,
    temperature: float,
) -> tuple[int, list[int], int]:
    active_nodes = {tree_idx: 0 for tree_idx in range(len(child_maps_batch))}
    active_paths = {tree_idx: [0] for tree_idx in range(len(child_maps_batch))}

    while True:
        representative_tree_idx = min(active_nodes)
        representative_node_idx = active_nodes[representative_tree_idx]
        next_token = int(
            sample(
                posterior_logits[
                    representative_tree_idx : representative_tree_idx + 1,
                    representative_node_idx : representative_node_idx + 1,
                ],
                temperature,
            )[0, 0]
        )

        next_active_nodes = {}
        next_active_paths = {}
        for tree_idx, node_idx in active_nodes.items():
            child_index = child_maps_batch[tree_idx][node_idx].get(next_token)
            if child_index is None:
                continue
            next_active_nodes[tree_idx] = child_index
            next_active_paths[tree_idx] = active_paths[tree_idx] + [child_index]

        if not next_active_nodes:
            selected_tree_idx = min(active_paths)
            return selected_tree_idx, active_paths[selected_tree_idx], next_token

        active_nodes = next_active_nodes
        active_paths = next_active_paths


@torch.inference_mode()
def pflash_v2_generate(
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
    branch_tree_budget = max(tree_budget // PFLASH_V2_BATCH_SIZE, 0)
    max_tree_nodes = 1 + branch_tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.empty(
        (PFLASH_V2_BATCH_SIZE, max_tree_nodes),
        dtype=torch.long,
        device=model.device,
    )
    verify_position_ids_buffer = torch.empty(
        (PFLASH_V2_BATCH_SIZE, max_tree_nodes),
        dtype=torch.long,
        device=model.device,
    )
    attention_mask_buffer = torch.zeros(
        (PFLASH_V2_BATCH_SIZE, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty(
        (PFLASH_V2_BATCH_SIZE, max_tree_nodes, max_tree_nodes),
        dtype=torch.bool,
        device=model.device,
    )

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(PFLASH_V2_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

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
    previous_tree_start = None

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = int(block_output_ids[0, 0])

        draft_stage_start = cuda_time()
        base_noise_embedding = target.model.embed_tokens(block_output_ids)
        noise_embedding_batch = build_perturbed_noise_embedding_batch(
            base_noise_embedding=base_noise_embedding,
            num_branches=PFLASH_V2_BATCH_SIZE,
            perturbation_temperature=perturbation_temperature,
            position_temperature_decay=position_temperature_decay,
        )
        projected_target_hidden = model.project_target_hidden(target_hidden)
        batched_target_hidden = projected_target_hidden.expand(PFLASH_V2_BATCH_SIZE, -1, -1)
        draft_position_ids = position_ids[
            :,
            past_key_values_draft.get_seq_length() : start + block_size,
        ].expand(PFLASH_V2_BATCH_SIZE, -1)
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
        trees = []
        for branch_idx in range(PFLASH_V2_BATCH_SIZE):
            (
                node_token_ids,
                node_depths,
                parents,
                child_maps,
                visibility_cpu,
                tree_build_subtimes,
            ) = build_ddtree_tree(draft_logits[branch_idx], branch_tree_budget)
            trees.append(SimpleNamespace(
                node_token_ids=node_token_ids.to(device=model.device, dtype=torch.long),
                node_depths=node_depths.to(device=model.device, dtype=torch.long),
                parents=parents,
                child_maps=child_maps,
                visibility_cpu=visibility_cpu,
            ))
            for stage_name, stage_elapsed in tree_build_subtimes.items():
                stage_times[stage_name] += stage_elapsed
        stage_times["tree_build"] += cuda_time() - tree_build_start

        tree_compile_start = cuda_time()
        (
            verify_input_ids,
            verify_position_ids,
            verify_attention_mask,
            current_lengths,
            previous_tree_start,
        ) = compile_batched_tree_group(
            root_token_id=root_token,
            start=start,
            trees=trees,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
        )
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        verify_past_key_values = repeat_dynamic_cache_batch(
            past_key_values=past_key_values_target,
            batch_size=PFLASH_V2_BATCH_SIZE,
        )
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=verify_past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        selected_tree_idx, accepted_indices, next_token = follow_verified_forest(
            child_maps_batch=[tree.child_maps for tree in trees],
            posterior_logits=output.logits,
            temperature=temperature,
        )
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        selected_verify_input_ids = verify_input_ids[selected_tree_idx : selected_tree_idx + 1]
        accepted_tokens = selected_verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        past_key_values_target = select_dynamic_cache_batch(
            past_key_values=verify_past_key_values,
            batch_index=selected_tree_idx,
        )
        compact_dynamic_cache(past_key_values_target, start, accepted_indices)

        target_hidden_batch = extract_context_feature(output.hidden_states, model.target_layer_ids)
        target_hidden = target_hidden_batch[selected_tree_idx : selected_tree_idx + 1].index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "selected_tree_idx": int(selected_tree_idx),
                "accepted_indices": [int(index) for index in accepted_indices],
                "branch_tree_budget": branch_tree_budget,
                "current_lengths": [int(length) for length in current_lengths],
                "trees": [
                    {
                        "node_token_ids": [int(token_id) for token_id in tree.node_token_ids.tolist()],
                        "node_depths": [int(depth) for depth in tree.node_depths.tolist()],
                        "parents": [int(parent) for parent in tree.parents],
                    }
                    for tree in trees
                ],
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
