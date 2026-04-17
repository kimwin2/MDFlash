from collections import Counter
import math
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from agreement_metrics import append_batch_agreement_metric, build_batch_agreement_snapshot
from dflash import cuda_time, dflash_generate, empty_stage_times
from model import DFlashDraftModel, extract_context_feature, sample
from pflash_v2 import repeat_dynamic_cache_batch, select_dynamic_cache_batch
from pflash_v7 import (
    PFLASH_V7_BATCH_SIZE,
    select_best_linear_branch,
    select_multiverse_anchor_tokens,
)


EXP_PREDICTMV_STAGE_ORDER = ("draft", "candidate_sample", "verify", "commit")


def _safe_zscores(values: list[float]) -> list[float]:
    if not values:
        return []
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    if variance <= 1e-12:
        return [0.0 for _ in values]
    std_value = math.sqrt(variance)
    return [(value - mean_value) / std_value for value in values]


def build_predictmv_metric(
    anchor_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    anchor_ranks: list[int | None],
    draft_logits: torch.Tensor,
    branch_acceptance_lengths: list[int],
    selected_branch_idx: int,
) -> dict[str, Any]:
    anchor_log_probs_full = torch.log_softmax(anchor_logits.float(), dim=-1)
    anchor_probs_full = anchor_log_probs_full.exp()
    selected_anchor_log_probs = anchor_log_probs_full.index_select(0, anchor_tokens)
    selected_anchor_probs = selected_anchor_log_probs.exp()
    selected_anchor_relative_probs = selected_anchor_probs / selected_anchor_probs.sum().clamp_min(1e-12)
    anchor_gap_to_best = selected_anchor_log_probs - selected_anchor_log_probs.max()

    anchor_topk = torch.topk(anchor_log_probs_full, k=min(2, anchor_log_probs_full.shape[0]), dim=-1).values
    anchor_top1_gap = float(anchor_topk[0] - anchor_topk[1]) if anchor_topk.numel() >= 2 else 0.0
    selected_anchor_topk = torch.topk(selected_anchor_log_probs, k=min(2, selected_anchor_log_probs.shape[0]), dim=-1).values
    selected_anchor_gap = float(selected_anchor_topk[0] - selected_anchor_topk[1]) if selected_anchor_topk.numel() >= 2 else 0.0
    anchor_entropy = float((-(anchor_probs_full * anchor_log_probs_full)).sum().item())
    selected_anchor_entropy = float(
        (-(selected_anchor_relative_probs * torch.log(selected_anchor_relative_probs.clamp_min(1e-12)))).sum().item()
    )

    draft_logits_float = draft_logits.float()
    draft_log_probs = torch.log_softmax(draft_logits_float, dim=-1)
    draft_probs = draft_log_probs.exp()
    draft_topk = torch.topk(draft_log_probs, k=min(2, draft_log_probs.shape[-1]), dim=-1)
    greedy_log_probs = draft_topk.values[..., 0]
    if draft_topk.values.shape[-1] >= 2:
        second_log_probs = draft_topk.values[..., 1]
    else:
        second_log_probs = greedy_log_probs
    greedy_probs = greedy_log_probs.exp()
    draft_margins = greedy_log_probs - second_log_probs
    draft_entropy = -(draft_probs * draft_log_probs).sum(dim=-1)
    candidate_token_ids = draft_topk.indices[..., 0].to(device="cpu", dtype=torch.long)
    agreement_snapshot = build_batch_agreement_snapshot(draft_logits)

    majority_tokens = agreement_snapshot["majority_tokens"] if agreement_snapshot is not None else []
    base_tokens = agreement_snapshot["base_tokens"] if agreement_snapshot is not None else []
    majority_agreement = agreement_snapshot["majority_agreement"] if agreement_snapshot is not None else []
    base_agreement = agreement_snapshot["base_agreement"] if agreement_snapshot is not None else []

    top1_by_branch = candidate_token_ids.tolist()
    branch_match_counts = [[0 for _ in range(len(top1_by_branch[0]))] for _ in range(len(top1_by_branch))]
    for depth_idx in range(len(top1_by_branch[0])):
        token_counts = Counter(int(branch_tokens[depth_idx]) for branch_tokens in top1_by_branch)
        for branch_idx, branch_tokens in enumerate(top1_by_branch):
            branch_match_counts[branch_idx][depth_idx] = int(token_counts[int(branch_tokens[depth_idx])])

    branch_features = []
    for branch_idx in range(PFLASH_V7_BATCH_SIZE):
        branch_unique_rate = sum(1 for count in branch_match_counts[branch_idx] if count == 1) / len(branch_match_counts[branch_idx])
        branch_non_unique_rate = sum(1 for count in branch_match_counts[branch_idx] if count > 1) / len(branch_match_counts[branch_idx])
        branch_majority_match_rate = (
            sum(
                1
                for depth_idx, token_id in enumerate(top1_by_branch[branch_idx])
                if depth_idx < len(majority_tokens) and int(token_id) == int(majority_tokens[depth_idx])
            )
            / len(top1_by_branch[branch_idx])
        )
        branch_base_match_rate = (
            sum(
                1
                for depth_idx, token_id in enumerate(top1_by_branch[branch_idx])
                if depth_idx < len(base_tokens) and int(token_id) == int(base_tokens[depth_idx])
            )
            / len(top1_by_branch[branch_idx])
        )

        branch_features.append({
            "branch_index": int(branch_idx),
            "anchor_token": int(anchor_tokens[branch_idx].item()),
            "anchor_rank": None if anchor_ranks[branch_idx] is None else int(anchor_ranks[branch_idx]),
            "anchor_logprob": float(selected_anchor_log_probs[branch_idx].item()),
            "anchor_prob": float(selected_anchor_probs[branch_idx].item()),
            "anchor_relative_prob": float(selected_anchor_relative_probs[branch_idx].item()),
            "anchor_gap_to_best": float(anchor_gap_to_best[branch_idx].item()),
            "draft_first_logprob": float(greedy_log_probs[branch_idx, 0].item()),
            "draft_mean_logprob": float(greedy_log_probs[branch_idx].mean().item()),
            "draft_sum_logprob": float(greedy_log_probs[branch_idx].sum().item()),
            "draft_min_logprob": float(greedy_log_probs[branch_idx].min().item()),
            "draft_last_logprob": float(greedy_log_probs[branch_idx, -1].item()),
            "draft_first_prob": float(greedy_probs[branch_idx, 0].item()),
            "draft_mean_prob": float(greedy_probs[branch_idx].mean().item()),
            "draft_first_margin": float(draft_margins[branch_idx, 0].item()),
            "draft_mean_margin": float(draft_margins[branch_idx].mean().item()),
            "draft_min_margin": float(draft_margins[branch_idx].min().item()),
            "draft_last_margin": float(draft_margins[branch_idx, -1].item()),
            "draft_first_entropy": float(draft_entropy[branch_idx, 0].item()),
            "draft_mean_entropy": float(draft_entropy[branch_idx].mean().item()),
            "draft_min_entropy": float(draft_entropy[branch_idx].min().item()),
            "draft_last_entropy": float(draft_entropy[branch_idx, -1].item()),
            "draft_majority_match_rate": float(branch_majority_match_rate),
            "draft_base_match_rate": float(branch_base_match_rate),
            "draft_unique_rate": float(branch_unique_rate),
            "draft_non_unique_rate": float(branch_non_unique_rate),
            "acceptance_length": int(branch_acceptance_lengths[branch_idx]),
        })

    predictor_scores = {
        "anchor_logprob": [feature["anchor_logprob"] for feature in branch_features],
        "anchor_prob": [feature["anchor_prob"] for feature in branch_features],
        "anchor_relative_prob": [feature["anchor_relative_prob"] for feature in branch_features],
        "anchor_gap_to_best": [feature["anchor_gap_to_best"] for feature in branch_features],
        "anchor_rank_score": [
            (-float(feature["anchor_rank"])) if feature["anchor_rank"] is not None else -1e9
            for feature in branch_features
        ],
        "draft_first_logprob": [feature["draft_first_logprob"] for feature in branch_features],
        "draft_mean_logprob": [feature["draft_mean_logprob"] for feature in branch_features],
        "draft_sum_logprob": [feature["draft_sum_logprob"] for feature in branch_features],
        "draft_min_logprob": [feature["draft_min_logprob"] for feature in branch_features],
        "draft_last_logprob": [feature["draft_last_logprob"] for feature in branch_features],
        "draft_first_prob": [feature["draft_first_prob"] for feature in branch_features],
        "draft_mean_prob": [feature["draft_mean_prob"] for feature in branch_features],
        "draft_first_margin": [feature["draft_first_margin"] for feature in branch_features],
        "draft_mean_margin": [feature["draft_mean_margin"] for feature in branch_features],
        "draft_min_margin": [feature["draft_min_margin"] for feature in branch_features],
        "draft_last_margin": [feature["draft_last_margin"] for feature in branch_features],
        "draft_first_neg_entropy": [-feature["draft_first_entropy"] for feature in branch_features],
        "draft_mean_neg_entropy": [-feature["draft_mean_entropy"] for feature in branch_features],
        "draft_min_neg_entropy": [-feature["draft_min_entropy"] for feature in branch_features],
        "draft_last_neg_entropy": [-feature["draft_last_entropy"] for feature in branch_features],
        "draft_majority_match_rate": [feature["draft_majority_match_rate"] for feature in branch_features],
        "draft_base_match_rate": [feature["draft_base_match_rate"] for feature in branch_features],
        "draft_non_unique_rate": [feature["draft_non_unique_rate"] for feature in branch_features],
        "draft_non_unique_mean_logprob": [
            feature["draft_non_unique_rate"] * feature["draft_mean_logprob"]
            for feature in branch_features
        ],
    }

    anchor_z = _safe_zscores(predictor_scores["anchor_logprob"])
    relative_prob_z = _safe_zscores(predictor_scores["anchor_relative_prob"])
    mean_logprob_z = _safe_zscores(predictor_scores["draft_mean_logprob"])
    first_logprob_z = _safe_zscores(predictor_scores["draft_first_logprob"])
    mean_margin_z = _safe_zscores(predictor_scores["draft_mean_margin"])
    majority_z = _safe_zscores(predictor_scores["draft_majority_match_rate"])
    predictor_scores["combo_anchor_mean_logprob"] = [a + b for a, b in zip(anchor_z, mean_logprob_z)]
    predictor_scores["combo_anchor_first_logprob"] = [a + b for a, b in zip(anchor_z, first_logprob_z)]
    predictor_scores["combo_anchor_mean_margin"] = [a + b for a, b in zip(anchor_z, mean_margin_z)]
    predictor_scores["combo_anchor_majority"] = [a + b for a, b in zip(anchor_z, majority_z)]
    predictor_scores["combo_relative_mean_logprob"] = [a + b for a, b in zip(relative_prob_z, mean_logprob_z)]

    sorted_acceptance = sorted((int(length) for length in branch_acceptance_lengths), reverse=True)
    best_acceptance = sorted_acceptance[0] if sorted_acceptance else 0
    second_acceptance = sorted_acceptance[1] if len(sorted_acceptance) >= 2 else best_acceptance

    return {
        "selected_branch": int(selected_branch_idx),
        "selected_anchor_rank": None if anchor_ranks[selected_branch_idx] is None else int(anchor_ranks[selected_branch_idx]),
        "branch_acceptance_lengths": [int(length) for length in branch_acceptance_lengths],
        "base_acceptance_length": int(branch_acceptance_lengths[0]),
        "selected_acceptance_length": int(branch_acceptance_lengths[selected_branch_idx]),
        "alternative_branch_selected": bool(selected_branch_idx != 0),
        "winner_margin": int(best_acceptance - second_acceptance),
        "anchor_tokens": [int(token_id) for token_id in anchor_tokens.tolist()],
        "anchor_ranks": [None if rank is None else int(rank) for rank in anchor_ranks],
        "branch_features": branch_features,
        "predictor_scores": predictor_scores,
        "context_features": {
            "anchor_entropy": float(anchor_entropy),
            "anchor_top1_gap": float(anchor_top1_gap),
            "selected_anchor_entropy": float(selected_anchor_entropy),
            "selected_anchor_gap": float(selected_anchor_gap),
            "selected_anchor_mass": float(selected_anchor_probs.sum().item()),
            "draft_mean_entropy_global": float(draft_entropy.mean().item()),
            "draft_mean_margin_global": float(draft_margins.mean().item()),
            "draft_mean_majority_agreement": (
                float(sum(majority_agreement) / len(majority_agreement))
                if majority_agreement
                else None
            ),
            "draft_mean_base_agreement": (
                float(sum(base_agreement) / len(base_agreement))
                if base_agreement
                else None
            ),
        },
    }


@torch.inference_mode()
def exp_predictmv_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
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

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.full(
        (PFLASH_V7_BATCH_SIZE, block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    verify_position_ids_buffer = position_ids[:, :block_size].expand(PFLASH_V7_BATCH_SIZE, -1).clone()

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(EXP_PREDICTMV_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    initial_root_token = sample(output.logits, temperature)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = initial_root_token
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    anchor_logits = output.logits[0, 0].detach()

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    batch_agreement_metrics = [] if measure_batch_agreement else None
    exp_predictmv_metrics = []
    draft_prefill = True

    while start < max_length:
        base_root_token = int(output_ids[0, start])
        round_anchor_logits = anchor_logits
        anchor_tokens, anchor_ranks = select_multiverse_anchor_tokens(
            anchor_logits=round_anchor_logits,
            branch0_token_id=base_root_token,
            num_branches=PFLASH_V7_BATCH_SIZE,
        )

        draft_stage_start = cuda_time()
        branch_block_output_ids = verify_input_ids_buffer
        branch_block_output_ids.fill_(mask_token_id)
        branch_block_output_ids[:, 0].copy_(anchor_tokens)

        draft_past_key_values = repeat_dynamic_cache_batch(
            past_key_values=past_key_values_draft,
            batch_size=PFLASH_V7_BATCH_SIZE,
        )
        projected_target_hidden = model.project_target_hidden(target_hidden)
        batched_target_hidden = projected_target_hidden.expand(PFLASH_V7_BATCH_SIZE, -1, -1)
        draft_position_ids = position_ids[
            :,
            past_key_values_draft.get_seq_length() : start + block_size,
        ].expand(PFLASH_V7_BATCH_SIZE, -1)
        draft_logits = target.lm_head(model(
            target_hidden=batched_target_hidden,
            target_hidden_is_projected=True,
            noise_embedding=target.model.embed_tokens(branch_block_output_ids),
            position_ids=draft_position_ids,
            past_key_values=draft_past_key_values,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_draft = select_dynamic_cache_batch(draft_past_key_values, 0)
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        sample_stage_start = cuda_time()
        candidate_token_ids = torch.argmax(draft_logits, dim=-1)
        if draft_horizon > 0:
            branch_block_output_ids[:, 1:] = candidate_token_ids
        verify_position_ids = verify_position_ids_buffer[:, :block_size]
        verify_position_ids.copy_(position_ids[:, start : start + block_size].expand(PFLASH_V7_BATCH_SIZE, -1))
        stage_times["candidate_sample"] += cuda_time() - sample_stage_start

        verify_stage_start = cuda_time()
        verify_past_key_values = repeat_dynamic_cache_batch(
            past_key_values=past_key_values_target,
            batch_size=PFLASH_V7_BATCH_SIZE,
        )
        output = target(
            branch_block_output_ids,
            position_ids=verify_position_ids,
            past_key_values=verify_past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        selected_branch_idx, accepted_indices, next_token, branch_acceptance_lengths = select_best_linear_branch(
            verify_input_ids=branch_block_output_ids,
            posterior=posterior,
        )
        append_batch_agreement_metric(batch_agreement_metrics, draft_logits, accepted_indices)
        accepted_length = len(accepted_indices)

        selected_tokens = branch_block_output_ids[selected_branch_idx : selected_branch_idx + 1, :accepted_length]
        output_ids[:, start : start + accepted_length] = selected_tokens
        output_ids[:, start + accepted_length] = next_token

        past_key_values_target = select_dynamic_cache_batch(verify_past_key_values, selected_branch_idx)
        past_key_values_target.crop(start + accepted_length)

        target_hidden_batch = extract_context_feature(output.hidden_states, model.target_layer_ids)
        target_hidden = target_hidden_batch[selected_branch_idx : selected_branch_idx + 1, :accepted_length, :]
        anchor_logits = output.logits[selected_branch_idx, accepted_length - 1].detach()

        round_metric = build_predictmv_metric(
            anchor_logits=round_anchor_logits,
            anchor_tokens=anchor_tokens,
            anchor_ranks=anchor_ranks,
            draft_logits=draft_logits,
            branch_acceptance_lengths=branch_acceptance_lengths,
            selected_branch_idx=selected_branch_idx,
        )
        exp_predictmv_metrics.append(round_metric)

        acceptance_lengths.append(accepted_length)
        start += accepted_length
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "selected_branch": int(selected_branch_idx),
                "analysis": round_metric,
                "candidate_token_ids": [
                    [int(token_id) for token_id in branch_tokens]
                    for branch_tokens in candidate_token_ids.tolist()
                ],
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - accepted_length : start + 1]
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
        exp_predictmv_metrics=exp_predictmv_metrics,
    )
