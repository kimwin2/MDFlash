from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from dflash import dflash_generate, cuda_time, empty_stage_times
from ddtree import (
    DDTREE_TREE_BUILD_STAGE_ORDER,
    build_ddtree_tree,
    compile_ddtree_tree,
    follow_verified_tree,
    compact_dynamic_cache,
)
from pexpress import build_perturbed_noise_embedding_batch
from agreement_metrics import append_batch_agreement_metric, build_batch_agreement_snapshot


PFLASH_V6_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")


def select_alignment_adaptive_config(
    draft_logits: torch.Tensor,
    max_block_size: int,
    max_tree_budget: int,
    high_alignment_threshold: float = 0.95,
    mid_alignment_threshold: float = 0.90,
    high_block_size: int = 16,
    mid_block_size: int = 8,
    low_block_size: int = 8,
    high_tree_budget: int = 128,
    mid_tree_budget: int = 64,
    low_tree_budget: int = 32,
) -> dict[str, Any]:
    snapshot = build_batch_agreement_snapshot(draft_logits)
    if snapshot is None:
        mean_alignment = 1.0
        first_alignment = 1.0
    else:
        majority_agreement = snapshot["majority_agreement"]
        mean_alignment = float(sum(majority_agreement) / len(majority_agreement))
        first_alignment = float(majority_agreement[0])

    if mean_alignment >= high_alignment_threshold:
        mode = "high"
        selected_block_size = high_block_size
        selected_tree_budget = high_tree_budget
    elif mean_alignment >= mid_alignment_threshold:
        mode = "mid"
        selected_block_size = mid_block_size
        selected_tree_budget = mid_tree_budget
    else:
        mode = "low"
        selected_block_size = low_block_size
        selected_tree_budget = low_tree_budget

    effective_block_size = max(2, min(int(selected_block_size), int(max_block_size)))
    effective_tree_budget = max(0, min(int(selected_tree_budget), int(max_tree_budget)))
    effective_horizon = max(1, min(effective_block_size - 1, int(draft_logits.shape[1])))

    return {
        "mode": mode,
        "mean_alignment": mean_alignment,
        "first_alignment": first_alignment,
        "effective_block_size": effective_block_size,
        "effective_horizon": effective_horizon,
        "effective_tree_budget": effective_tree_budget,
    }


@torch.inference_mode()
def pflash_v6_generate(
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
    high_alignment_threshold: float = 0.95,
    mid_alignment_threshold: float = 0.90,
    high_block_size: int = 16,
    mid_block_size: int = 8,
    low_block_size: int = 8,
    high_tree_budget: int = 128,
    mid_tree_budget: int = 64,
    low_tree_budget: int = 32,
    measure_batch_agreement: bool = False,
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
    stage_times = empty_stage_times(PFLASH_V6_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

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
    batch_agreement_metrics = [] if measure_batch_agreement else None
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]
        num_branches = max(tree_budget // block_size, 2)

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
        adaptive_config = select_alignment_adaptive_config(
            draft_logits=draft_logits,
            max_block_size=block_size,
            max_tree_budget=tree_budget,
            high_alignment_threshold=high_alignment_threshold,
            mid_alignment_threshold=mid_alignment_threshold,
            high_block_size=high_block_size,
            mid_block_size=mid_block_size,
            low_block_size=low_block_size,
            high_tree_budget=high_tree_budget,
            mid_tree_budget=mid_tree_budget,
            low_tree_budget=low_tree_budget,
        )

        node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes = build_ddtree_tree(
            draft_logits[0, : adaptive_config["effective_horizon"]],
            adaptive_config["effective_tree_budget"],
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
        append_batch_agreement_metric(batch_agreement_metrics, draft_logits, accepted_indices)
        if batch_agreement_metrics:
            batch_agreement_metrics[-1].update({
                "adaptive_mode": adaptive_config["mode"],
                "effective_block_size": adaptive_config["effective_block_size"],
                "effective_tree_budget": adaptive_config["effective_tree_budget"],
            })
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
                "adaptive": {
                    "mode": adaptive_config["mode"],
                    "mean_alignment": float(adaptive_config["mean_alignment"]),
                    "first_alignment": float(adaptive_config["first_alignment"]),
                    "effective_block_size": int(adaptive_config["effective_block_size"]),
                    "effective_tree_budget": int(adaptive_config["effective_tree_budget"]),
                },
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
        batch_agreement_metrics=batch_agreement_metrics,
    )
