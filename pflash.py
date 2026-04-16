import heapq
import math
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


PFLASH_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")


def build_branch_log_priors(
    num_branches: int,
    perturbation_temperature: float,
    branch_prior_weight: float,
    device: torch.device,
) -> torch.Tensor:
    if num_branches <= 0:
        raise ValueError("num_branches must be positive.")
    if perturbation_temperature < 0.0:
        raise ValueError("perturbation_temperature must be non-negative.")
    if branch_prior_weight < 0.0:
        raise ValueError("branch_prior_weight must be non-negative.")

    if num_branches == 1 or perturbation_temperature < 1e-5 or branch_prior_weight < 1e-5:
        return torch.zeros((num_branches,), device=device, dtype=torch.float32)

    branch_positions = torch.linspace(0.0, 1.0, steps=num_branches, device=device, dtype=torch.float32)
    return -(branch_prior_weight * branch_positions.square())


def build_pflash_tree(
    draft_logits: torch.Tensor,
    budget: int,
    branch_log_priors: torch.Tensor | None = None,
    merge_prefix_branches: bool = False,
    prefix_support_bonus_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor, dict[str, float]]:
    if prefix_support_bonus_weight < 0.0:
        raise ValueError("prefix_support_bonus_weight must be non-negative.")
    if merge_prefix_branches:
        return build_merged_prefix_pflash_tree(
            draft_logits=draft_logits,
            budget=budget,
            branch_log_priors=branch_log_priors,
            prefix_support_bonus_weight=prefix_support_bonus_weight,
        )

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
        )

    num_branches, depth_limit, vocab_size = draft_logits.shape
    topk = min(budget, vocab_size)

    copy_start = cuda_time()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    if branch_log_priors is None:
        branch_log_priors_cpu = torch.zeros((num_branches,), dtype=torch.float32)
    else:
        branch_log_priors_cpu = branch_log_priors.to(device="cpu", dtype=torch.float32)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()
    branch_log_priors_np = branch_log_priors_cpu.numpy()

    heap_start = time.perf_counter()
    heap: list[tuple[float, int, tuple[int, ...], int, int, float]] = []
    for branch_idx in range(num_branches):
        first_logw = float(branch_log_priors_np[branch_idx] + top_log_probs_np[branch_idx, 0, 0])
        heapq.heappush(heap, (-first_logw, branch_idx, (0,), 1, 0, first_logw))

    parents = [-1]
    child_maps: list[dict[int, int]] = [dict()]
    node_token_ids: list[int] = []
    node_depths: list[int] = []
    tree_full = False

    while heap and len(node_token_ids) < budget:
        _, branch_idx, ranks, depth, rank, logw = heapq.heappop(heap)
        prefix_tokens = tuple(
            int(top_token_ids_np[branch_idx, prefix_depth, prefix_rank])
            for prefix_depth, prefix_rank in enumerate(ranks)
        )

        current_index = 0
        for prefix_depth, token_id in enumerate(prefix_tokens, start=1):
            child_index = child_maps[current_index].get(token_id)
            if child_index is None:
                if len(node_token_ids) >= budget:
                    tree_full = True
                    break
                child_index = len(parents)
                parents.append(current_index)
                child_maps.append(dict())
                child_maps[current_index][token_id] = child_index
                node_token_ids.append(token_id)
                node_depths.append(prefix_depth)
            current_index = child_index
        if tree_full:
            break

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs_np[branch_idx, depth - 1, rank]) + float(
                top_log_probs_np[branch_idx, depth - 1, rank + 1]
            )
            heapq.heappush(
                heap,
                (-sibling_logw, branch_idx, sibling_ranks, depth, rank + 1, sibling_logw),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[branch_idx, depth, 0])
            heapq.heappush(
                heap,
                (-child_logw, branch_idx, child_ranks, depth + 1, 0, child_logw),
            )

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
    )


def build_merged_prefix_pflash_tree(
    draft_logits: torch.Tensor,
    budget: int,
    branch_log_priors: torch.Tensor | None = None,
    prefix_support_bonus_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor, dict[str, float]]:
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
        )

    num_branches, depth_limit, vocab_size = draft_logits.shape
    topk = min(budget, vocab_size)

    copy_start = cuda_time()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    if branch_log_priors is None:
        branch_log_priors_cpu = torch.zeros((num_branches,), dtype=torch.float32)
    else:
        branch_log_priors_cpu = branch_log_priors.to(device="cpu", dtype=torch.float32)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()
    branch_log_priors_np = branch_log_priors_cpu.numpy()

    heap_start = time.perf_counter()
    heap: list[tuple[float, tuple[int, ...], int]] = []
    candidate_states: list[tuple[int, tuple[int, ...], int, int, float]] = []
    prefix_entries: dict[tuple[int, ...], SimpleNamespace] = {}

    def build_prefix_tokens(branch_idx: int, ranks: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            int(top_token_ids_np[branch_idx, prefix_depth, prefix_rank])
            for prefix_depth, prefix_rank in enumerate(ranks)
        )

    def aggregate_score(entry: SimpleNamespace) -> float:
        support_count = len(entry.supporting_branches)
        return float(entry.representative_logw + prefix_support_bonus_weight * math.log1p(support_count))

    def push_prefix_entry(prefix_tokens: tuple[int, ...], entry: SimpleNamespace) -> None:
        heapq.heappush(heap, (-aggregate_score(entry), prefix_tokens, entry.version))

    def should_promote_representative(
        entry: SimpleNamespace,
        branch_idx: int,
        logw: float,
        state_id: int,
    ) -> bool:
        representative_branch_idx = entry.representative_branch_idx
        if branch_idx == 0 and representative_branch_idx != 0:
            return True
        if branch_idx != 0 and representative_branch_idx == 0:
            return False
        if logw > entry.representative_logw:
            return True
        if logw < entry.representative_logw:
            return False
        return state_id < entry.representative_state_id

    def add_candidate_state(
        branch_idx: int,
        ranks: tuple[int, ...],
        depth: int,
        rank: int,
        logw: float,
    ) -> None:
        prefix_tokens = build_prefix_tokens(branch_idx, ranks)
        state_id = len(candidate_states)
        candidate_states.append((branch_idx, ranks, depth, rank, logw))

        entry = prefix_entries.get(prefix_tokens)
        if entry is None:
            entry = SimpleNamespace(
                supporting_branches={branch_idx},
                pending_state_ids=[state_id],
                in_tree=False,
                representative_state_id=state_id,
                representative_branch_idx=branch_idx,
                representative_logw=logw,
                version=0,
            )
            prefix_entries[prefix_tokens] = entry
        else:
            entry.supporting_branches.add(branch_idx)
            if entry.in_tree:
                return
            entry.pending_state_ids.append(state_id)
            if should_promote_representative(entry, branch_idx, logw, state_id):
                entry.representative_state_id = state_id
                entry.representative_branch_idx = branch_idx
                entry.representative_logw = logw
            entry.version += 1

        push_prefix_entry(prefix_tokens, entry)

    for branch_idx in range(num_branches):
        first_logw = float(branch_log_priors_np[branch_idx] + top_log_probs_np[branch_idx, 0, 0])
        add_candidate_state(
            branch_idx=branch_idx,
            ranks=(0,),
            depth=1,
            rank=0,
            logw=first_logw,
        )

    parents = [-1]
    child_maps: list[dict[int, int]] = [dict()]
    node_token_ids: list[int] = []
    node_depths: list[int] = []
    tree_full = False

    while heap and len(node_token_ids) < budget:
        _, prefix_tokens, version = heapq.heappop(heap)
        entry = prefix_entries.get(prefix_tokens)
        if entry is None or version != entry.version or not entry.pending_state_ids:
            continue

        entry.pending_state_ids = []
        entry.version += 1

        if not entry.in_tree:
            current_index = 0
            for prefix_depth, token_id in enumerate(prefix_tokens, start=1):
                child_index = child_maps[current_index].get(token_id)
                if child_index is None:
                    if len(node_token_ids) >= budget:
                        tree_full = True
                        break
                    child_index = len(parents)
                    parents.append(current_index)
                    child_maps.append(dict())
                    child_maps[current_index][token_id] = child_index
                    node_token_ids.append(token_id)
                    node_depths.append(prefix_depth)
                current_index = child_index
            entry.in_tree = True
        if tree_full:
            break

        representative_state_id = entry.representative_state_id
        branch_idx, ranks, depth, rank, logw = candidate_states[representative_state_id]

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs_np[branch_idx, depth - 1, rank]) + float(
                top_log_probs_np[branch_idx, depth - 1, rank + 1]
            )
            add_candidate_state(
                branch_idx=branch_idx,
                ranks=sibling_ranks,
                depth=depth,
                rank=rank + 1,
                logw=sibling_logw,
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[branch_idx, depth, 0])
            add_candidate_state(
                branch_idx=branch_idx,
                ranks=child_ranks,
                depth=depth + 1,
                rank=0,
                logw=child_logw,
            )

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
    )


@torch.inference_mode()
def pflash_generate(
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
    branch_prior_weight: float = 0.5,
    merge_prefix_branches: bool = False,
    prefix_support_bonus_weight: float = 0.0,
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
    stage_times = empty_stage_times(PFLASH_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

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
        num_branches = max(tree_budget // block_size, 1)

        draft_stage_start = cuda_time()
        base_noise_embedding = target.model.embed_tokens(block_output_ids)
        noise_embedding_batch = build_perturbed_noise_embedding_batch(
            base_noise_embedding=base_noise_embedding,
            num_branches=num_branches,
            perturbation_temperature=perturbation_temperature,
            position_temperature_decay=position_temperature_decay,
        )
        projected_target_hidden = model.project_target_hidden(target_hidden)
        batched_target_hidden = projected_target_hidden.expand(num_branches, -1, -1)
        branch_log_priors = build_branch_log_priors(
            num_branches=num_branches,
            perturbation_temperature=perturbation_temperature,
            branch_prior_weight=branch_prior_weight,
            device=model.device,
        )
        draft_position_ids = position_ids[
            :,
            past_key_values_draft.get_seq_length() : start + block_size,
        ].expand(num_branches, -1)
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
        node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes = build_pflash_tree(
            draft_logits=draft_logits,
            budget=tree_budget,
            branch_log_priors=branch_log_priors,
            merge_prefix_branches=merge_prefix_branches,
            prefix_support_bonus_weight=prefix_support_bonus_weight,
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
